"""Revue visuelle de la détection sur les captures d'un match : planches d'images avec ce que voit le détecteur.

Chaque image est numérotée ; les unités sont encadrées (bleu = nous, rouge = eux) avec nom et confiance.
Sert à repérer à l'œil les erreurs (ratées, fantômes, ombres, mauvais camp) et à noter les corrections.

  .venv-yolo\\Scripts\\python scripts/detect_review.py            (dernier match, 16 images)
  .venv-yolo\\Scripts\\python scripts/detect_review.py 20260927-084128 --n 16
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("game", nargs="?")
    ap.add_argument("--n", type=int, default=16)
    a = ap.parse_args()
    import sys
    sys.path.insert(0, str(ROOT))
    from clashai import battle as B
    from clashai.detect_yolo import Detector
    games = sorted(Path(p).name for p in glob.glob(str(ROOT / "runs/capture/*")))
    game = a.game or games[-1]
    fs = [f for f in sorted(glob.glob(str(ROOT / f"runs/capture/{game}/*.jpg")))]
    fs = [f for f in fs if B.in_battle(cv2.imread(f))]
    pick = [fs[int(i)] for i in np.linspace(0, len(fs) - 1, min(a.n, len(fs)))]
    det = Detector(track=False, conf=0.35)
    out_dir = ROOT / "runs/detector/review" / game
    out_dir.mkdir(parents=True, exist_ok=True)
    tiles = []
    for k, f in enumerate(pick):
        img = cv2.imread(f)
        crop, _ = det._crop(img)
        units = det.on_arena(crop)
        v = crop.copy()
        for u in units:
            col = (40, 40, 235) if u.enemy else (255, 150, 30)
            cv2.rectangle(v, u.box[:2], u.box[2:], col, 2)
            cv2.putText(v, f"{u.name[:12]} {u.conf:.0%}", (u.box[0], max(10, u.box[1] - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(v, f"#{k} {Path(f).stem}", (6, 890), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        tiles.append(v)
        (out_dir / f"{k:02d}.txt").write_text(Path(f).name)
    for s in range(0, len(tiles), 4):
        row = tiles[s:s + 4] + [np.zeros_like(tiles[0])] * (4 - len(tiles[s:s + 4]))
        cv2.imwrite(str(out_dir / f"sheet_{s // 4}.jpg"), cv2.resize(cv2.hconcat(row), (1600, 630)))
    print(f"{len(tiles)} images -> {out_dir}")


if __name__ == "__main__":
    main()
