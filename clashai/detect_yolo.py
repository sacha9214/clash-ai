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

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "models/yolo/clashai_yolo11s.engine"      # repli : le .pt à côté si TensorRT est absent
# Recadrage de l'arène dans l'écran (fractions x0, y0, largeur, hauteur), comme pour KataCR
ARENA = (0.020, 0.035, 0.960, 0.684)
ARENA_SIZE = (568, 896)
IMGSZ = 896                     # remplacé par models/yolo/clashai_yolo11s.json si le modèle adopté en demande une autre
UI = {"bar", "bar-level", "tower-bar", "king-tower-bar", "dagger-duchess-tower-bar", "elixir",
      "clock", "emote", "evolution-symbol", "ice-spirit-evolution-symbol", "text", "selected"}


@dataclass
class Unit:
    track_id: int          # -1 si non suivi
    name: str
    enemy: bool
    conf: float
    box: tuple[int, int, int, int]   # x0, y0, x1, y1 dans l'image du flux
    coasted: bool = False            # pas vue sur cette image : gardée de mémoire (courte disparition)

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
        self.memory: dict[int, tuple[Unit, float, float, float]] = {}   # suivi -> (unité, instant, vx, vy en px/s)
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

    def _crop(self, frame: np.ndarray) -> tuple[np.ndarray, tuple[float, float, float, float]]:
        h, w = frame.shape[:2]
        x0, y0 = int(ARENA[0] * w), int(ARENA[1] * h)
        x1, y1 = int((ARENA[0] + ARENA[2]) * w), int((ARENA[1] + ARENA[3]) * h)
        crop = cv2.resize(frame[y0:y1, x0:x1], ARENA_SIZE, interpolation=cv2.INTER_LINEAR)
        return crop, (x0, y0, (x1 - x0) / ARENA_SIZE[0], (y1 - y0) / ARENA_SIZE[1])

    def __call__(self, frame: np.ndarray) -> list[Unit]:
        crop, offset = self._crop(frame)
        return self.on_arena(crop, offset)

    def on_arena(self, crop: np.ndarray, offset=(0, 0, 1.0, 1.0)) -> list[Unit]:
        """Détecte sur une arène déjà recadrée en 568x896 ; offset replace les boîtes dans l'image source."""
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
            if name in UI or name.startswith("padding") or (self.track and conf < 0.35):
                continue
            box = (int(ox + x0 * sx), int(oy + y0 * sy), int(ox + x1 * sx), int(oy + y1 * sy))
            enemy = side == "1"
            if tid >= 0:
                # le camp d une unité ne change jamais : vote sur TOUTES ses images depuis son apparition
                v = self.side_votes.setdefault(int(tid), collections.deque(maxlen=None))   # tout son historique : figé
                if len(v) < 3:
                    v.append(conf if enemy else -conf)          # décidé sur ses 3 premières images…
                enemy = sum(v) > 0                              # …puis figé pour toute sa vie
            units.append(Unit(int(tid), name, enemy, float(conf), box))
        if self.track:
            units = self._coast(units)
        self._update_trails([u for u in units if not u.coasted])
        return units

    COAST_S = 0.4

    def _coast(self, units: list[Unit]) -> list[Unit]:
        now = time.perf_counter()
        seen = {u.track_id for u in units if u.track_id >= 0}
        for u in units:
            if u.track_id < 0:
                continue
            old = self.memory.get(u.track_id)
            vx = vy = 0.0
            if old:
                dt = max(now - old[1], 1e-3)
                vx = 0.5 * old[2] + 0.5 * (u.center[0] - old[0].center[0]) / dt     # vitesse lissée
                vy = 0.5 * old[3] + 0.5 * (u.center[1] - old[0].center[1]) / dt
            self.memory[u.track_id] = (u, now, vx, vy)
        out = list(units)
        for tid, (u, t, vx, vy) in list(self.memory.items()):
            if tid in seen:
                continue
            age = now - t
            if age > self.COAST_S or "tower" in u.name:
                if age > 2.0:
                    del self.memory[tid]
                continue
            dx, dy = int(vx * age), int(vy * age)
            out.append(Unit(tid, u.name, u.enemy, u.conf * 0.9, (u.box[0] + dx, u.box[1] + dy, u.box[2] + dx, u.box[3] + dy),
                            coasted=True))
        return out

    def _update_trails(self, units: list[Unit]):
        alive = set()
        for u in units:
            if u.track_id >= 0:
                alive.add(u.track_id)
                self.trails.setdefault(u.track_id, collections.deque(maxlen=20)).append(u.center)
        for tid in list(self.trails):
            if tid not in alive:
                del self.trails[tid]
        for tid in list(self.side_votes):
            if tid not in alive and len(self.side_votes) > 200:
                del self.side_votes[tid]


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
