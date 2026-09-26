"""Détection + suivi des unités (phase 2a) avec les détecteurs pré-entraînés de KataCR.

KataCR (MIT, github.com/wty-yy/KataCR) a entraîné deux YOLOv8 modifiés (sortie
« camp » en plus de la classe) sur ~150 types d'objets. Ils demandent l'ancien
ultralytics 8.1 : ce module tourne donc dans .venv-katacr.

Les images sont recadrées sur l'arène et redimensionnées en 568x896 comme à
l'entraînement, puis les boîtes sont replacées dans les coordonnées du flux.
"""
from __future__ import annotations

import collections
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/KataCR"))
# ultralytics 8.1 charge ses .pt par pickle complet ; torch >= 2.6 (requis pour les RTX 50xx) le refuse
# par défaut. Modèles KataCR de confiance -> on garde l'ancien comportement.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import torch  # noqa: E402
import torchvision  # noqa: E402
from katacr.constants.label_list import idx2unit, unit2idx  # noqa: E402
from katacr.yolov8.custom_result import CRResults  # noqa: E402
from katacr.yolov8.custom_trackers import cr_on_predict_postprocess_end, cr_on_predict_start  # noqa: E402
from katacr.yolov8.train import YOLO_CR  # noqa: E402

DETECTORS = [ROOT / "models/katacr/detector1_v0.7.13.pt", ROOT / "models/katacr/detector2_v0.7.13.pt"]
TRACKER_CFG = ROOT / "third_party/KataCR/katacr/yolov8/bytetrack.yaml"
# Recadrage de l'arène dans l'écran (fractions x0, y0, largeur, hauteur) : même
# proportion 568x896 que les images d'entraînement de KataCR.
ARENA = (0.020, 0.035, 0.960, 0.684)
ARENA_SIZE = (568, 896)
# Éléments d'interface détectés mais inutiles à afficher comme « unités »
UI = {"bar", "bar-level", "tower-bar", "king-tower-bar", "dagger-duchess-tower-bar", "elixir",
      "clock", "emote", "evolution-symbol", "ice-spirit-evolution-symbol"}
UI |= {n for n in idx2unit.values() if n.startswith("padding")}


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
                 weights: list | None = None):
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.half = self.device != "cpu"
        self.models = [YOLO_CR(str(p)) for p in (weights or DETECTORS)]   # weights : nos propres modèles
        self.iou = iou
        self.tracker = None
        self.conf = conf
        if track:
            self.conf = 0.1                  # ByteTrack exploite aussi les détections faibles
            self.tracker_cfg_path = str(TRACKER_CFG)
            cr_on_predict_start(self, persist=True)   # crée self.tracker
        self.trails: dict[int, collections.deque] = {}

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
        preds = []
        for m in self.models:
            r = m.predict(crop, verbose=False, conf=self.conf, device=self.device, half=self.half, imgsz=896)[0]
            b = r.orig_boxes.clone().cpu()
            for i in range(len(b)):
                b[i, 5] = unit2idx[r.names[int(b[i, 5])]]   # indices propres à chaque détecteur -> communs
            preds.append(b)
        preds = torch.cat(preds, 0) if preds else torch.zeros(0, 7)
        preds = preds[torchvision.ops.nms(preds[:, :4], preds[:, 4], self.iou)]
        self.result = CRResults(crop, path="", names=idx2unit, boxes=preds)
        if self.tracker is not None:
            cr_on_predict_postprocess_end(self, persist=True)
        data = self.result.get_data()   # xyxy, (track_id), conf, cls, bel
        units = []
        for row in data:
            tid = int(row[4]) if row.shape[0] == 8 else -1
            conf, cls, bel = row[-3], int(row[-2]), int(row[-1])
            name = idx2unit[cls]
            if name in UI or (self.tracker is not None and conf < 0.35):
                continue
            box = (int(ox + row[0] * sx), int(oy + row[1] * sy), int(ox + row[2] * sx), int(oy + row[3] * sy))
            units.append(Unit(tid, name, bool(bel), float(conf), box))
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


if __name__ == "__main__":
    # Test hors ligne : python -m clashai.detect image.png [sortie.png]
    img = cv2.imread(sys.argv[1])
    det = Detector(track=False)
    det(img)
    t = time.perf_counter()
    units = det(img)
    print(f"{(time.perf_counter() - t) * 1000:.0f} ms, {len(units)} unités")
    for u in units:
        print(f"  {'ennemi' if u.enemy else 'allié ':6} {u.name:20} {u.conf:.2f} {u.box}")
    draw(img, units)
    cv2.imwrite(sys.argv[2] if len(sys.argv) > 2 else "detect.png", img)
