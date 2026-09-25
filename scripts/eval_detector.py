"""Note objective d'un détecteur sur les images réelles annotées à la main de KataCR.

Pour chaque image (arène 568x896) : prédictions vs vérité (classe + boîte, IoU >= 0.5).
Rapporte précision, rappel, F1, camp correct (bleu/rouge), vitesse, et les classes les plus ratées.
Même seuil de confiance qu'en match (0.5, sans suivi).

  .venv-katacr\\Scripts\\python scripts/eval_detector.py                       # détecteurs KataCR
  .venv-katacr\\Scripts\\python scripts/eval_detector.py runs/detector/x/weights/best.pt
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from clashai import detect as D  # noqa: E402
from katacr.constants.label_list import idx2unit  # noqa: E402

VAL = ROOT / "data/Clash-Royale-Dataset/images/part2"
N_IMAGES = 600          # sous-ensemble fixe, réparti sur toutes les séquences


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def truth(img_path: Path, w: int, h: int) -> list[tuple[str, int, tuple]]:
    out = []
    for line in open(img_path.with_suffix(".txt")):
        p = line.split()
        name = idx2unit[int(p[0])]
        if name in D.UI or name in ("text", "selected"):   # pas des unités : l'IA les ignore
            continue
        cx, cy, bw, bh = (float(v) for v in p[1:5])
        out.append((name, int(p[5]), ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)))
    return out


def main():
    weights = [a for a in sys.argv[1:] if a.endswith(".pt")] or None
    det = D.Detector(track=False, weights=weights)
    files = [VAL / l.strip() for l in open(VAL / "yolo_annotation.txt") if l.strip()]
    files = files[:: max(1, len(files) // N_IMAGES)][:N_IMAGES]
    tp = fp = fn = side_ok = 0
    missed, confused = collections.Counter(), collections.Counter()
    times = []
    for f in files:
        img = cv2.imread(str(f))
        crop = cv2.resize(img, D.ARENA_SIZE)
        sx, sy = img.shape[1] / D.ARENA_SIZE[0], img.shape[0] / D.ARENA_SIZE[1]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        preds = det.on_arena(crop)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
        gt = truth(f, img.shape[1], img.shape[0])
        used = set()
        preds = [u for u in preds if u.name not in ("text", "selected")]
        for u in sorted(preds, key=lambda u: -u.conf):
            box = (u.box[0] * sx, u.box[1] * sy, u.box[2] * sx, u.box[3] * sy)
            best, j = 0.5, None
            for k, (name, bel, g) in enumerate(gt):
                if k not in used and name == u.name and iou(box, g) >= best:
                    best, j = iou(box, g), k
            if j is None:
                fp += 1
                near = [n for n, _, g in gt if iou(box, g) >= 0.5]
                if near:
                    confused[(near[0], u.name)] += 1      # bonne place, mauvaise classe
            else:
                used.add(j)
                tp += 1
                side_ok += int(gt[j][1] == int(u.enemy))
        for k, (name, _, _) in enumerate(gt):
            if k not in used:
                fn += 1
                missed[name] += 1
    p, r = tp / max(1, tp + fp), tp / max(1, tp + fn)
    res = {"model": weights or "KataCR detector1+2", "images": len(files), "precision": round(p, 3),
           "recall": round(r, 3), "f1": round(2 * p * r / max(1e-9, p + r), 3),
           "side_accuracy": round(side_ok / max(1, tp), 3), "ms_per_image": round(float(np.median(times[5:])), 1),
           "most_missed": missed.most_common(10),
           "most_confused": [f"{a} -> {b} x{k}" for (a, b), k in confused.most_common(8)]}
    print(json.dumps(res, ensure_ascii=False, indent=1))
    out = ROOT / "runs/detector/eval.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(res, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
