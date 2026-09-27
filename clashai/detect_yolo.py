"""Détection + suivi des unités avec NOTRE détecteur : YOLO11s entraîné sur le GPU, exporté en TensorRT (FP16).

Entraîné la nuit du 25/09 (scripts/night_detector.py) sur 60 000 images synthétiques + 4 199 réelles.
Sur des séquences jamais vues : F1 0.939 (KataCR : 0.916), rappel 90.5 % (86.8 %), camp 100 %,
8.5 ms/image (28.4 ms). Classes « unité_camp » : knight_0 = le nôtre (bas), knight_1 = ennemi.

Même interface que detect_katacr.py (Unit, Detector, draw, ARENA…) : le reste de l'IA ne change pas.
Tourne dans .venv-yolo (ultralytics 8.4, torch cu128, TensorRT).
"""
from __future__ import annotations

import collections
import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from clashai.cards import DECK
from clashai.identity import TrackIdentity
from clashai.motion import Tracks

# nos unités possibles (deck exact + évolutions) : une unité à nous ne peut pas porter un autre nom
OWN_NAMES = {u for c in DECK.values() for u in c.units} | {u + "-evolution" for c in DECK.values() for u in c.units}

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "models/yolo/clashai_yolo11s.engine"      # repli : le .pt à côté si TensorRT est absent
# Recadrage de l'arène dans l'écran (fractions x0, y0, largeur, hauteur), comme pour KataCR
ARENA = (0.020, 0.035, 0.960, 0.684)
ARENA_SIZE = (568, 896)
DECOR_Y = 45                    # px de l'arène recadrée : au-dessus, décor seulement (cristaux pris pour une Tesla)
IMGSZ = 896                     # remplacé par models/yolo/clashai_yolo11s.json si le modèle adopté en demande une autre
UI = {"bar", "bar-level", "tower-bar", "king-tower-bar", "dagger-duchess-tower-bar", "skeleton-king-bar", "elixir",
      "clock", "emote", "evolution-symbol", "ice-spirit-evolution-symbol", "text", "selected"}


@dataclass
class Unit:
    track_id: int          # -1 si non suivi
    name: str
    enemy: bool
    conf: float
    box: tuple[int, int, int, int]   # x0, y0, x1, y1 dans l'image du flux
    coasted: bool = False            # pas vue sur cette image : gardée de mémoire (courte disparition)
    # vitesse en px/s de l'image du flux (x vers la droite, y vers le bas), du milieu du bas de la boîte, mesurée sur
    # l'horloge réelle (clashai/motion.py) ; (0, 0) si non suivie ou trop récente ; de mémoire si coasted
    vel: tuple[float, float] = (0.0, 0.0)

    @property
    def center(self) -> tuple[int, int]:
        return (self.box[0] + self.box[2]) // 2, (self.box[1] + self.box[3]) // 2


class Detector:
    def __init__(self, device: str | None = None, track: bool = True, conf: float = 0.5, iou: float = 0.6,
                 weights: list | str | None = None):
        w = Path(weights[0] if isinstance(weights, list) else weights) if weights else WEIGHTS
        if not w.exists():
            w = w.with_suffix(".pt")
        self.model = YOLO(str(w), task="detect")
        side = w.with_suffix(".json")
        self.imgsz = json.loads(side.read_text())["imgsz"] if side.exists() else IMGSZ
        self.device = device or 0
        self.track, self.iou = track, iou
        self.conf = 0.1 if track else conf      # ByteTrack exploite aussi les détections faibles
        self.trails: dict[int, collections.deque] = {}
        self.side_votes: dict[int, collections.deque] = {}   # suivi -> votes récents (+conf ennemi, -conf allié)
        # mémoire courte : une unité suivie qui disparaît moins de COAST_S secondes (petite unité ratée sur une
        # image, masquée par un effet) est gardée à sa position prévue au lieu de « clignoter »
        self.memory: dict[int, tuple[Unit, float]] = {}   # suivi -> (dernière unité vue, avec sa vitesse ; instant)
        self.motion = Tracks()
        self.ident: dict[int, TrackIdentity] = {}   # suivi -> camp et nom votés sur toute sa vie                    # suivi -> vitesse mesurée sur ses positions horodatées
        self.seen_count: dict[int, int] = {}      # suivi -> nombre d'images où il a été vu
        self.confirmed: set[int] = set()          # suivis confirmés (plus une ombre)
        self._fresh = True
        self.tracker = self                      # interface de detect_katacr : det.tracker.reset()
        self.on_arena(np.zeros((ARENA_SIZE[1], ARENA_SIZE[0], 3), np.uint8))   # chauffe (TensorRT)
        self.reset()

    def reset(self) -> None:
        """Nouveau combat : le suivi repart de zéro."""
        self._fresh = True
        self.trails.clear()
        self.side_votes.clear()
        self.memory.clear()
        self.motion.clear()
        self.ident.clear()
        self.seen_count.clear()
        self.confirmed.clear()

    def _crop(self, frame: np.ndarray) -> tuple[np.ndarray, tuple[float, float, float, float]]:
        h, w = frame.shape[:2]
        x0, y0 = int(ARENA[0] * w), int(ARENA[1] * h)
        x1, y1 = int((ARENA[0] + ARENA[2]) * w), int((ARENA[1] + ARENA[3]) * h)
        crop = cv2.resize(frame[y0:y1, x0:x1], ARENA_SIZE, interpolation=cv2.INTER_LINEAR)
        return crop, (x0, y0, (x1 - x0) / ARENA_SIZE[0], (y1 - y0) / ARENA_SIZE[1])

    def __call__(self, frame: np.ndarray, t: float | None = None) -> list[Unit]:
        crop, offset = self._crop(frame)
        return self.on_arena(crop, offset, t)

    def on_arena(self, crop: np.ndarray, offset=(0, 0, 1.0, 1.0), t: float | None = None) -> list[Unit]:
        """Détecte sur une arène déjà recadrée en 568x896 ; offset replace les boîtes dans l'image source.
        t : instant de l'image en s (vidéo, tests) ; par défaut l'horloge au moment de l'appel."""
        now = time.perf_counter() if t is None else t
        ox, oy, sx, sy = offset
        kw = dict(imgsz=self.imgsz, conf=self.conf, iou=self.iou, verbose=False, device=self.device)   # moteur TensorRT déjà en FP16
        if self.track:
            r = self.model.track(crop, persist=not self._fresh, tracker="bytetrack.yaml", **kw)[0]
            self._fresh = False
        else:
            r = self.model.predict(crop, **kw)[0]
        b = r.boxes
        ids = b.id.int().tolist() if b.id is not None else [-1] * len(b)
        units = []
        for (x0, y0, x1, y1), c, conf, tid in zip(b.xyxy.tolist(), b.cls.int().tolist(), b.conf.tolist(), ids):
            name, _, side = r.names[c].rpartition("_")
            if name in UI or name.startswith("padding") or (self.track and conf < 0.2):
                continue
            if (y0 + y1) / 2 < DECOR_Y and "tower" not in name:
                continue        # bande du décor au-dessus du Roi ennemi : aucune troupe n'y va (Arène 4 : « Tesla »)
            box = (int(ox + x0 * sx), int(oy + y0 * sy), int(ox + x1 * sx), int(oy + y1 * sy))
            enemy = side == "1"
            color = team_color(crop, (int(x0), int(y0), int(x1), int(y1))) if "tower" not in name else 0
            if tid >= 0 and "tower" not in name:
                # camp et nom votés sur toute la vie du suivi (clashai/identity.py) : lieu de naissance, badge, détecteur
                idt = self.ident.get(int(tid))
                if idt is None:
                    idt = self.ident[int(tid)] = TrackIdentity(y1 / crop.shape[0])
                idt.observe(name, conf, enemy, color, now)
                enemy = idt.enemy
                name = idt.name(None if enemy else OWN_NAMES)
            elif color:
                enemy = color > 0
            units.append(Unit(int(tid), name, enemy, float(conf), box))
        if len(self.ident) > 300:
            self.ident = {k: v for k, v in self.ident.items() if now - v.last_t < 5}
        if self.track:
            self.motion.forget(now)
            self.motion.observe(units, now)      # avant _confirm : la 1re image d'une unité compte aussi
            units = self._coast(self._confirm(units), now)
        self._update_trails([u for u in units if not u.coasted])
        return self._drop_tower_ghosts(units)

    @staticmethod
    def _drop_tower_ghosts(units: list[Unit]) -> list[Unit]:
        """Une « unité » dont la boîte recouvre presque une tour est la tour elle-même mal lue (ex. Chevalier sur notre
        tour princesse) : on l'écarte. Une vraie unité qui attaque une tour est plus petite ou décalée."""
        towers = [u.box for u in units if "tower" in u.name]

        def iou(a, b):
            ix = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
            return ix / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix + 1e-9)
        return [u for u in units if "tower" in u.name or not any(iou(u.box, t) > 0.45 for t in towers)]

    COAST_S = 0.8
    CONFIRM_CONF = 0.6          # une unité NOUVELLE doit être sûre (>= 60 %)…
    CONFIRM_FRAMES = 2          # …ou vue sur 2 images de suite : une ombre ou un effet ne passe pas

    def _confirm(self, units: list[Unit]) -> list[Unit]:
        """Ombres, effets de sorts, reflets : détections faibles et fugaces. Une unité n'est utilisée que si elle est
        sûre, ou confirmée sur plusieurs images ; une fois confirmée, on la garde même si sa confiance baisse
        (petite unité dans une mêlée) : on perd moins d'unités réelles."""
        out = []
        for u in units:
            if u.track_id < 0 or "tower" in u.name:
                out.append(u)
                continue
            n = self.seen_count.get(u.track_id, 0) + 1
            self.seen_count[u.track_id] = n
            if u.track_id in self.confirmed or u.conf >= self.CONFIRM_CONF or (n >= self.CONFIRM_FRAMES and u.conf >= 0.35):
                self.confirmed.add(u.track_id)
                out.append(u)
        if len(self.seen_count) > 2000:
            self.seen_count.clear()
        return out

    def _coast(self, units: list[Unit], now: float) -> list[Unit]:
        seen = {u.track_id for u in units if u.track_id >= 0}
        for u in units:
            if u.track_id >= 0:
                self.memory[u.track_id] = (u, now)
        out = list(units)
        for tid, (u, t) in list(self.memory.items()):
            if tid in seen:
                continue
            age = now - t
            if age > self.COAST_S or "tower" in u.name:
                if age > 2.0:
                    del self.memory[tid]
                continue
            # position prévue avec la dernière vitesse mesurée, gardée telle quelle
            dx, dy = round(u.vel[0] * age), round(u.vel[1] * age)
            out.append(Unit(tid, u.name, u.enemy, u.conf * 0.9, (u.box[0] + dx, u.box[1] + dy, u.box[2] + dx, u.box[3] + dy),
                            coasted=True, vel=u.vel))
        return out

    def _update_trails(self, units: list[Unit]):
        """Traces des unités vues ; celle d'une unité en mémoire (coasted) est gardée jusqu'à ce qu'elle soit oubliée."""
        for u in units:
            if u.track_id >= 0:
                self.trails.setdefault(u.track_id, collections.deque(maxlen=20)).append(u.center)
        alive = set(self.memory) | {u.track_id for u in units}
        for tid in list(self.trails):
            if tid not in alive:
                del self.trails[tid]
        for tid in list(self.side_votes):
            if tid not in alive and len(self.side_votes) > 200:
                del self.side_votes[tid]


def team_color(img: np.ndarray, box) -> int:
    """Camp lu sur la couleur du badge de niveau / de la barre de vie au-dessus de l'unité (bleu = nous, rouge = eux).
    +1 ennemi, -1 allié, 0 illisible. Sur nos captures : juste dans 5 désaccords sur 5 avec le détecteur."""
    x0, y0, x1, y1 = box
    h = y1 - y0
    # strictement AU-DESSUS de l'unité (badge de niveau + barre de vie) : pas sur l'unité, où l'horloge rouge du
    # déploiement ferait croire à un ennemi
    ya, yb = max(0, y0 - int(0.35 * h) - 8), max(1, y0 + int(0.05 * h))
    xa, xb = max(0, x0 - 4), min(img.shape[1], x1 + 4)
    if yb <= ya or xb <= xa:
        return 0
    hsv = cv2.cvtColor(img[ya:yb, xa:xb], cv2.COLOR_BGR2HSV)
    strong = (hsv[..., 1] > 120) & (hsv[..., 2] > 120)
    red = int((((hsv[..., 0] < 8) | (hsv[..., 0] > 170)) & strong).sum())
    blue = int((((hsv[..., 0] > 95) & (hsv[..., 0] < 125)) & strong).sum())
    if max(red, blue) < 12 or min(red, blue) > 0.5 * max(red, blue):
        return 0
    return 1 if red > blue else -1


BLUE, RED = (255, 150, 30), (40, 40, 235)


def draw(img: np.ndarray, units: list[Unit], trails: dict | None = None) -> None:
    for u in units:
        col = RED if u.enemy else BLUE
        if trails and u.track_id in trails and len(trails[u.track_id]) > 1:
            pts = np.array(trails[u.track_id], np.int32)
            cv2.polylines(img, [pts], False, col, 2, cv2.LINE_AA)
        x0, y0, x1, y1 = u.box
        cv2.rectangle(img, (x0, y0), (x1, y1), col, 2)
        label = (f"#{u.track_id} " if u.track_id >= 0 else "") + f"{u.name} {u.conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.rectangle(img, (x0, y0 - th - 6), (x0 + tw + 4, y0), col, -1)
        cv2.putText(img, label, (x0 + 2, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
