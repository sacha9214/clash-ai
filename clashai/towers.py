"""État du match que le cerveau ne voyait pas : PV de nos tours, tours ennemies détruites, phase du match.

- PV de nos tours princesses : longueur de la barre bleue sous le chiffre (continue depuis son début,
  pour ignorer ce qui passe devant). Mesuré sur le REDMAGIC (578x1280) : 1180/1750 -> 0.66, 912/1750 -> 0.50.
- Tours ennemies détruites : la tour princesse n'est plus détectée pendant plus de 3 s (à la place : des gravats).
  (Leurs PV exacts : barre rouge à calibrer sur des captures brutes, les captures annotées la cachent.)
- Phase : le chrono se déduit du début du combat (2:00 -> double élixir, 3:00 -> prolongations).
"""
from __future__ import annotations

import collections
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

# Barres de PV de nos tours princesses (fractions de l'écran) : début, fin, rangée
OUR_BARS = {0: (0.168, 0.285, 0.592), 1: (0.756, 0.874, 0.592)}
DOUBLE_ELIXIR_S, OVERTIME_S = 120.0, 180.0
# tours princesses et leurs variantes (troupes de tour) : sinon, contre un Canonnier, les deux tours ennemies
# passaient pour détruites au bout de 3 s
PRINCESS_TOWERS = {"queen-tower", "cannoneer-tower", "dagger-duchess-tower"}
CALIB_S = 20.0               # début du match : tours ennemies intactes -> longueur d'une barre pleine
DESTROYED_AFTER_S = 6.0      # barre basse puis illisible aussi longtemps : la tour est tombée
CROWN_AFTER_S = 6.0          # tour ennemie absente aussi longtemps : couronne pour nous (3 s ne suffit pas pour décider
                             # de ne plus attaquer : une tour masquée par des unités ne doit pas nous faire croire en tête)


def bar_fraction(img: np.ndarray, x0f: float, x1f: float, yf: float) -> float | None:
    """Part remplie d'une barre bleue (0-1), ou None si quelque chose la cache."""
    h, w = img.shape[:2]
    x0, x1, y = int(x0f * w), int(x1f * w), int(yf * h)
    hsv = cv2.cvtColor(img[y - 1:y + 2, x0:x1 + 3], cv2.COLOR_BGR2HSV)
    blue = ((hsv[..., 0] > 90) & (hsv[..., 0] < 110) & (hsv[..., 1] > 80) & (hsv[..., 2] > 150)).any(0)
    if not blue[:4].any():
        return None
    start = int(np.argmax(blue[:4]))                  # la barre peut commencer 1-2 px plus loin selon le côté
    rest = blue[start:]
    run = int(np.argmin(rest)) if not rest.all() else len(rest)   # segment continu depuis le début de la barre
    return min(1.0, run / (x1 - x0 - start))


def red_run(img: np.ndarray, box) -> int | None:
    """Longueur (px) de la barre de PV rouge au-dessus d'une tour ennemie (boîte du détecteur) : la plus longue suite de
    pixels rouges sur une rangée de la bande au-dessus de la tour. None si rien de rouge."""
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    h, w = img.shape[:2]
    xa, xb = max(0, int(x0 - 0.3 * bw)), min(w, int(x1 + 0.3 * bw))
    ya, yb = max(0, int(y0 - 0.45 * bh)), min(h, int(y0 + 0.25 * bh))
    if xb - xa < 8 or yb - ya < 2:
        return None
    hsv = cv2.cvtColor(img[ya:yb, xa:xb], cv2.COLOR_BGR2HSV)
    red = ((hsv[..., 0] < 8) | (hsv[..., 0] > 170)) & (hsv[..., 1] > 100) & (hsv[..., 2] > 120)
    best = 0
    for row in red:
        # plus longue suite de True sur la rangée
        d = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
        starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        if len(starts):
            best = max(best, int((ends - starts).max()))
    return best or None


@dataclass
class MatchState:
    start: float = field(default_factory=time.time)
    our_hp: dict = field(default_factory=lambda: {0: 1.0, 1: 1.0})          # couloir -> fraction de PV
    _hp_hist: dict = field(default_factory=lambda: {0: collections.deque(maxlen=5), 1: collections.deque(maxlen=5)})
    enemy_alive: dict = field(default_factory=lambda: {0: True, 1: True})   # tours princesses ennemies
    _enemy_seen: dict = field(default_factory=lambda: {0: time.time(), 1: time.time()})
    _bar_seen: dict = field(default_factory=lambda: {0: None, 1: None})   # dernier instant où la barre a été lue
    # PV des tours princesses ENNEMIES : longueur de la barre rouge au-dessus de chaque tour, rapportée à sa longueur
    # pleine mesurée pendant les CALIB_S premières secondes (tours intactes). Utilisés seulement si cette longueur est
    # nette et stable (enemy_hp_ok) : sinon la fonction reste coupée (pas de sort sur une tour mal lue)
    enemy_hp: dict = field(default_factory=lambda: {0: 1.0, 1: 1.0})
    enemy_hp_ok: bool = False
    _full_runs: dict = field(default_factory=lambda: {0: [], 1: []})
    _enemy_hist: dict = field(default_factory=lambda: {0: collections.deque(maxlen=9), 1: collections.deque(maxlen=9)})

    def update(self, img: np.ndarray, units, now: float | None = None) -> None:
        now = time.time() if now is None else now
        h0, w0 = img.shape[:2]
        for u in units:
            if u.name not in PRINCESS_TOWERS or u.center[1] >= 0.4 * h0 or getattr(u, "box", None) is None:
                continue                                  # seulement les tours du haut (ennemies)
            run = red_run(img, u.box)
            lane = 0 if u.center[0] < w0 / 2 else 1
            if self.elapsed(now) < CALIB_S:
                if run:
                    self._full_runs[lane].append(run)
                continue
            full = self._full_length()
            if full is None:
                continue
            hist = self._enemy_hist[lane]
            hist.append(min(1.0, (run or 0) / full))
            # médiane glissante : une unité qui passe devant la barre (lecture trop basse) ne la « tue » pas
            self.enemy_hp[lane] = float(np.median(hist))
        self.enemy_hp_ok = self._full_length() is not None
        for lane, (x0, x1, y) in OUR_BARS.items():
            f = bar_fraction(img, x0, x1, y)
            if f is not None:
                self._bar_seen[lane] = now
                self._hp_hist[lane].append(f)
                self.our_hp[lane] = float(np.median(self._hp_hist[lane]))   # médiane : robuste aux passages devant
            elif (self._bar_seen[lane] is not None and self.our_hp[lane] < 0.3
                  and now - self._bar_seen[lane] > DESTROYED_AFTER_S):
                self.our_hp[lane] = 0.0          # tour détruite : sa barre a disparu (avant : restait à sa dernière valeur)
        h, w = img.shape[:2]
        for u in units:
            if u.name in PRINCESS_TOWERS and u.center[1] < 0.4 * h:
                self._enemy_seen[0 if u.center[0] < w / 2 else 1] = now
        for lane in (0, 1):
            self.enemy_alive[lane] = now - self._enemy_seen[lane] < 3.0

    def elapsed(self, now: float | None = None) -> float:
        return (time.time() if now is None else now) - self.start

    def phase(self, now: float | None = None) -> str:
        t = self.elapsed(now)
        return "normal" if t < DOUBLE_ELIXIR_S else "double" if t < OVERTIME_S else "overtime"

    def _full_length(self) -> float | None:
        """Longueur d'une barre pleine, si elle a été mesurée nettement (assez de lectures, toutes proches)."""
        runs = self._full_runs[0] + self._full_runs[1]
        if len(runs) < 6:
            return None
        med = float(np.median(runs))
        spread = np.percentile(runs, 90) - np.percentile(runs, 10)
        return med if med >= 15 and spread <= 0.15 * med else None

    def crowns(self, now: float | None = None) -> tuple[int, int]:
        """(nos couronnes, les siennes) : ses tours princesses tombées, les nôtres à 0 PV."""
        now = time.time() if now is None else now
        ours = sum(now - self._enemy_seen[lane] > CROWN_AFTER_S for lane in (0, 1))
        theirs = sum(self.our_hp[lane] <= 0.0 for lane in (0, 1))
        return ours, theirs

    def summary(self) -> str:
        down = [("G", "D")[l] for l in (0, 1) if not self.enemy_alive[l]]
        return (f"{self.phase()} {int(self.elapsed()) // 60}:{int(self.elapsed()) % 60:02d}  nos tours "
                f"{self.our_hp[0]:.0%}/{self.our_hp[1]:.0%}" + (f"  tour ennemie détruite : {','.join(down)}" if down else ""))
