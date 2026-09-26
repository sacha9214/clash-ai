"""Pseudo-étiquetage des nouvelles cartes avec le détecteur v2 (entraîné sur les rendus du Fan Kit).

Pour chaque image en attente (vidéo ciblée sur une nouvelle carte, indice dans <nom>.hint) : le modèle v2 cherche
cette carte ; s'il la voit avec assez de confiance, l'image est enregistrée avec ses boîtes (celles du modèle pour
tout le reste). Sinon elle reste en attente. Les images retenues sont listées pour vérification à l'œil
(runs/detector/pseudo_labels.json) avant de s'en servir pour un nouvel entraînement.

  .venv-yolo\\Scripts\\python scripts/pseudo_label.py --model runs/detector/yolo11s_cr_v2/weights/best.pt
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
REAL = Path("D:/clash-ai-dataset/real")
MIN_CONF = 0.55


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "runs/detector/yolo11s_cr_v2/weights/best.pt"))
    a = ap.parse_args()
    model = YOLO(a.model)
    classes = json.loads((REAL / "classes.json").read_text(encoding="utf-8"))
    cid = {n: i for i, n in enumerate(classes)}
    kept, report = 0, []
    for hint in sorted((REAL / "pending").glob("*.hint")):
        card = hint.read_text().strip()
        img_f = hint.with_suffix(".jpg")
        if not img_f.exists():
            continue
        r = model.predict(cv2.imread(str(img_f)), imgsz=896, conf=0.25, verbose=False, device=0)[0]
        found = [(r.names[c], float(p)) for c, p in zip(r.boxes.cls.int().tolist(), r.boxes.conf.tolist())
                 if r.names[c].rsplit("_", 1)[0] == card and p >= MIN_CONF]
        if not found:
            continue
        lines = []
        for (x, y, w, h), c, p in zip(r.boxes.xywhn.tolist(), r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
            name = r.names[c]
            if p < 0.4:
                continue
            if name not in cid:                   # classe connue du modèle v2 mais pas encore de l'outil
                classes.append(name)
                cid[name] = len(classes) - 1
            lines.append(f"{cid[name]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
        stem = "pseudo_" + img_f.stem
        (REAL / "labels").mkdir(parents=True, exist_ok=True)
        (REAL / "images").mkdir(parents=True, exist_ok=True)
        (REAL / "labels" / f"{stem}.txt").write_text("\n".join(lines))
        shutil.move(img_f, REAL / "images" / f"{stem}.jpg")
        for ext in (".json", ".hint"):
            img_f.with_suffix(ext).unlink(missing_ok=True)
        report.append({"image": stem, "card": card, "conf": round(max(p for _, p in found), 2)})
        kept += 1
    (REAL / "classes.json").write_text(json.dumps(classes, ensure_ascii=False, indent=0), encoding="utf-8")
    (ROOT / "runs/detector/pseudo_labels.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    by = {}
    for x in report:
        by[x["card"]] = by.get(x["card"], 0) + 1
    print(f"{kept} images pseudo-étiquetées : {by}")


if __name__ == "__main__":
    main()
