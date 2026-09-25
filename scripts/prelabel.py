"""Phase C du détecteur : pré-étiquette nos propres captures de match avec le détecteur actuel.

Les captures brutes (runs/capture/<match>/*.jpg, écran entier du téléphone) sont recadrées sur
l'arène (568x896, comme KataCR) ; le détecteur propose ses boîtes ; il ne reste qu'à corriger ses
erreurs dans scripts/label_tool.py. Une image sur `--every` pour éviter les quasi-doublons.

  .venv-katacr\\Scripts\\python scripts/prelabel.py --every 2
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from clashai import detect as D  # noqa: E402
from clashai import battle as B  # noqa: E402

OUT = Path("D:/clash-ai-dataset/real")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=2)
    a = ap.parse_args()
    (OUT / "pending").mkdir(parents=True, exist_ok=True)
    det = D.Detector(track=False)
    done = {p.stem for p in (OUT / "pending").glob("*.jpg")} | {p.stem for p in (OUT / "images").glob("*.jpg")}
    n = 0
    for game in sorted(glob.glob(str(ROOT / "runs/capture/*"))):
        for i, f in enumerate(sorted(glob.glob(game + "/*.jpg"))):
            stem = f"{Path(game).name}_{Path(f).stem}"
            if i % a.every or stem in done:
                continue
            frame = cv2.imread(f)
            if frame is None or not B.in_battle(frame):
                continue
            crop, _ = det._crop(frame)
            units = det.on_arena(crop)
            aw, ah = D.ARENA_SIZE
            boxes = [{"name": u.name, "side": int(u.enemy), "conf": round(u.conf, 2),
                      "box": [round(u.box[0] / aw, 4), round(u.box[1] / ah, 4), round(u.box[2] / aw, 4), round(u.box[3] / ah, 4)]}
                     for u in units if u.name not in ("text", "selected")]
            cv2.imwrite(str(OUT / "pending" / f"{stem}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            (OUT / "pending" / f"{stem}.json").write_text(json.dumps(boxes))
            n += 1
    print(f"{n} images pré-étiquetées -> {OUT / 'pending'}")


if __name__ == "__main__":
    main()
