"""Étiquette les captures de NOS matchs (runs/capture/*) avec le détecteur actuel, en ne gardant que ses
détections sûres (conf >= --conf). Donne au modèle des images de NOTRE écran (graphismes récents, résolution du
téléphone) pour l'entraînement v2. Une planche de contrôle est écrite pour vérification à l'œil.

  .venv-yolo\\Scripts\\python scripts/label_captures.py --conf 0.7
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("D:/clash-ai-dataset")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.7)
    a = ap.parse_args()
    import sys
    sys.path.insert(0, str(ROOT))
    from clashai import battle as B
    from clashai.detect_yolo import Detector
    det = Detector(track=False, conf=a.conf)
    names_v2 = yaml.safe_load(open(DATA / "data_v2.yaml"))["names"]
    cid = {n: i for i, n in names_v2.items()}
    (DATA / "images/train_fk").mkdir(parents=True, exist_ok=True)
    (DATA / "labels/train_fk").mkdir(parents=True, exist_ok=True)
    n = boxes = 0
    sheet = []
    for f in sorted(glob.glob(str(ROOT / "runs/capture/*/*.jpg"))):
        frame = cv2.imread(f)
        if frame is None or not B.in_battle(frame):
            continue
        crop, _ = det._crop(frame)
        r = det.model.predict(crop, imgsz=896, conf=a.conf, verbose=False, device=0)[0]
        lines = []
        for (x, y, w, h), c in zip(r.boxes.xywhn.tolist(), r.boxes.cls.int().tolist()):
            name = r.names[c]
            if name in cid:
                lines.append(f"{cid[name]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
        if len(lines) < 4:                     # que les tours : peu utile
            continue
        stem = f"cap_{Path(f).parent.name}_{Path(f).stem}"
        cv2.imwrite(str(DATA / "images/train_fk" / f"{stem}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (DATA / "labels/train_fk" / f"{stem}.txt").write_text("\n".join(lines))
        n += 1
        boxes += len(lines)
        if n % 60 == 1 and len(sheet) < 6:
            v = r.plot(labels=True, conf=False, line_width=1)
            sheet.append(cv2.resize(v, (284, 448)))
    if sheet:
        cv2.imwrite(str(ROOT / "runs/detector/captures_check.jpg"), cv2.hconcat(sheet))
    print(f"{n} captures de nos matchs étiquetées ({boxes} boîtes, conf >= {a.conf})")


if __name__ == "__main__":
    main()
