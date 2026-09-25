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


@dataclass
class MatchState:
    start: float = field(default_factory=time.time)
    our_hp: dict = field(default_factory=lambda: {0: 1.0, 1: 1.0})          # couloir -> fraction de PV
    _hp_hist: dict = field(default_factory=lambda: {0: collections.deque(maxlen=5), 1: collections.deque(maxlen=5)})
    enemy_alive: dict = field(default_factory=lambda: {0: True, 1: True})   # tours princesses ennemies
    _enemy_seen: dict = field(default_factory=lambda: {0: time.time(), 1: time.time()})

    def update(self, img: np.ndarray, units, now: float | None = None) -> None:
        now = time.time() if now is None else now
        for lane, (x0, x1, y) in OUR_BARS.items():
            f = bar_fraction(img, x0, x1, y)
            if f is not None:
                self._hp_hist[lane].append(f)
                self.our_hp[lane] = float(np.median(self._hp_hist[lane]))   # médiane : robuste aux passages devant
        h, w = img.shape[:2]
        for u in units:
            if u.name == "queen-tower" and u.center[1] < 0.4 * h:
                self._enemy_seen[0 if u.center[0] < w / 2 else 1] = now
        for lane in (0, 1):
            self.enemy_alive[lane] = now - self._enemy_seen[lane] < 3.0

    def elapsed(self, now: float | None = None) -> float:
        return (time.time() if now is None else now) - self.start

    def phase(self, now: float | None = None) -> str:
        t = self.elapsed(now)
        return "normal" if t < DOUBLE_ELIXIR_S else "double" if t < OVERTIME_S else "overtime"

    def summary(self) -> str:
        down = [("G", "D")[l] for l in (0, 1) if not self.enemy_alive[l]]
        return (f"{self.phase()} {int(self.elapsed()) // 60}:{int(self.elapsed()) % 60:02d}  nos tours "
                f"{self.our_hp[0]:.0%}/{self.our_hp[1]:.0%}" + (f"  tour ennemie détruite : {','.join(down)}" if down else ""))
