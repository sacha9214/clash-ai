"""Même note que eval_detector.py, pour un modèle ultralytics récent (classes « unité_camp »).

Mêmes 600 images réelles, même seuil (0.5), même appariement : les scores sont comparables.

  .venv-yolo\\Scripts\\python scripts/eval_yolo.py runs/detector/yolo11s_cr/weights/best.pt
"""
from __future__ import annotations

import collections
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/KataCR"))
from katacr.constants.label_list import idx2unit  # noqa: E402

VAL = ROOT / "data/Clash-Royale-Dataset/images/part2"
TEST_LIST = Path("D:/clash-ai-dataset/test_real.txt")
ARENA_SIZE = (568, 896)
SKIP = {"bar", "bar-level", "tower-bar", "king-tower-bar", "dagger-duchess-tower-bar", "elixir", "clock", "emote",
        "evolution-symbol", "ice-spirit-evolution-symbol", "text", "selected"}


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def skip(name: str) -> bool:
    return name in SKIP or name.startswith("padding")


def main():
    weights = sys.argv[1]
    model = YOLO(weights)
    # test : séquences vidéo ENTIÈRES jamais vues à l'entraînement (D:/clash-ai-dataset/test_real.txt),
    # pour que des images voisines presque identiques ne faussent pas la note
    files = [VAL / l.strip() for l in open(TEST_LIST) if l.strip()]
    tp = fp = fn = side_ok = 0
    missed, confused, times = collections.Counter(), collections.Counter(), []
    for f in files:
        img = cv2.imread(str(f))
        crop = cv2.resize(img, ARENA_SIZE)
        sx, sy = img.shape[1] / ARENA_SIZE[0], img.shape[0] / ARENA_SIZE[1]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = model.predict(crop, imgsz=896, conf=0.5, half=True, verbose=False, device=0)[0]
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
        preds = []
        for (x0, y0, x1, y1), c, conf in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            name, bel = r.names[int(c)].rsplit("_", 1)
            if not skip(name):
                preds.append((name, int(bel), conf, (x0 * sx, y0 * sy, x1 * sx, y1 * sy)))
        gt = []
        for line in open(f.with_suffix(".txt")):
            p = line.split()
            name = idx2unit[int(p[0])]
            if skip(name):
                continue
            cx, cy, bw, bh = (float(v) for v in p[1:5])
            W, H = img.shape[1], img.shape[0]
            gt.append((name, int(p[5]), ((cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H)))
        used = set()
        for name, bel, conf, box in sorted(preds, key=lambda p: -p[2]):
            best, j = 0.5, None
            for k, (gname, gbel, g) in enumerate(gt):
                if k not in used and gname == name and iou(box, g) >= best:
                    best, j = iou(box, g), k
            if j is None:
                fp += 1
                near = [n for n, _, g in gt if iou(box, g) >= 0.5]
                if near:
                    confused[(near[0], name)] += 1
            else:
                used.add(j)
                tp += 1
                side_ok += int(gt[j][1] == bel)
        for k, (name, _, _) in enumerate(gt):
            if k not in used:
                fn += 1
                missed[name] += 1
    p, rc = tp / max(1, tp + fp), tp / max(1, tp + fn)
    res = {"model": weights, "images": len(files), "precision": round(p, 3), "recall": round(rc, 3),
           "f1": round(2 * p * rc / max(1e-9, p + rc), 3), "side_accuracy": round(side_ok / max(1, tp), 3),
           "ms_per_image": round(float(np.median(times[5:])), 1), "most_missed": missed.most_common(10),
           "most_confused": [f"{a} -> {b} x{k}" for (a, b), k in confused.most_common(8)]}
    print(json.dumps(res, ensure_ascii=False, indent=1))
    with open(ROOT / "runs/detector/eval.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(res, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
