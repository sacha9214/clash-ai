"""Détection + suivi des unités avec NOTRE détecteur : YOLO11s entraîné sur le GPU, exporté en TensorRT (FP16).

Entraîné la nuit du 25/09 (scripts/night_detector.py) sur 60 000 images synthétiques + 4 199 réelles.
Sur des séquences jamais vues : F1 0.939 (KataCR : 0.916), rappel 90.5 % (86.8 %), camp 100 %,
8.5 ms/image (28.4 ms). Classes « unité_camp » : knight_0 = le nôtre (bas), knight_1 = ennemi.

Même interface que detect_katacr.py (Unit, Detector, draw, ARENA…) : le reste de l'IA ne change pas.
Tourne dans .venv-yolo (ultralytics 8.4, torch cu128, TensorRT).
"""
from __future__ import annotations

import collections
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
IMGSZ = 896
UI = {"bar", "bar-level", "tower-bar", "king-tower-bar", "dagger-duchess-tower-bar", "elixir",
      "clock", "emote", "evolution-symbol", "ice-spirit-evolution-symbol", "text", "selected"}


@dataclass
class Unit:
    track_id: int          # -1 si non suivi
    name: str
    enemy: bool
    conf: float
    box: tuple[int, int, int, int]   # x0, y0, x1, y1 dans l'image du flux

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
        self.device = device or 0
        self.track, self.iou = track, iou
        self.conf = 0.1 if track else conf      # ByteTrack exploite aussi les détections faibles
        self.trails: dict[int, collections.deque] = {}
        self._fresh = True
        self.tracker = self                      # interface de detect_katacr : det.tracker.reset()
        self.on_arena(np.zeros((ARENA_SIZE[1], ARENA_SIZE[0], 3), np.uint8))   # chauffe (TensorRT)
        self.reset()

    def reset(self) -> None:
        """Nouveau combat : le suivi repart de zéro."""
        self._fresh = True
        self.trails.clear()

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
        kw = dict(imgsz=IMGSZ, conf=self.conf, iou=self.iou, verbose=False, device=self.device)   # moteur TensorRT déjà en FP16
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
            units.append(Unit(int(tid), name, side == "1", float(conf), box))
        self._update_trails(units)
        return units

    def _update_trails(self, units: list[Unit]):
        alive = set()
        for u in units:
            if u.track_id >= 0:
                alive.add(u.track_id)
                self.trails.setdefault(u.track_id, collections.deque(maxlen=20)).append(u.center)
        for tid in list(self.trails):
            if tid not in alive:
                del self.trails[tid]


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
