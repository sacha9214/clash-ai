"""Réglage fin COURT du détecteur (~25 min) sur nos propres captures, entre deux séries de matchs.

- Données : les images de NOS matchs étiquetées d'après nos poses (own_labels.py), 20 % des matchs gardés
  de côté pour le test « notre écran », plus un échantillon des images habituelles pour ne rien oublier.
- Test 1 « notre écran » : sur les matchs mis de côté, part de nos unités posées que le détecteur voit.
- Test 2 « cartes connues » : le test habituel (séquences KataCR jamais vues).
- Adopté seulement si le test 1 monte et que le test 2 ne baisse pas de plus de 0.005.

  .venv-yolo\\Scripts\\python scripts/quick_finetune.py --minutes 25
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("D:/clash-ai-dataset")
MODELS = ROOT / "models/yolo"
PY = ROOT / ".venv-yolo/Scripts/python.exe"
QUICK = DATA / "quick"


def is_test_game(stem: str) -> bool:
    game = "_".join(stem.split("_")[1:3])
    return int(hashlib.md5(game.encode()).hexdigest(), 16) % 5 == 0


def own_recall(model_path: Path, imgsz: int, stems: list[str]) -> float:
    """Part des unités NOUS (étiquetées d'après nos poses) retrouvées par le modèle (même classe, IoU >= 0.3)."""
    from ultralytics import YOLO
    names = yaml.safe_load(open(DATA / "data_v2.yaml"))["names"]
    m = YOLO(str(model_path), task="detect")
    ours = {"archer", "knight", "valkyrie", "mini-pekka", "giant", "cannon"}
    found = total = 0
    for s in stems:
        img = DATA / "images/train_fk" / f"{s}.jpg"
        gt = [l.split() for l in (DATA / "labels/train_fk" / f"{s}.txt").read_text().splitlines() if l.strip()]
        gt = [(names[int(c)], *map(float, b)) for c, *b in gt if names[int(c)].rsplit("_", 1)[0] in ours
              and names[int(c)].endswith("_0")]
        if not gt:
            continue
        r = m.predict(str(img), imgsz=imgsz, conf=0.35, verbose=False, device=0)[0]
        pred = [(r.names[int(c)], *b) for c, b in zip(r.boxes.cls.tolist(), r.boxes.xywhn.tolist())]
        for n, x, y, w, h in gt:
            total += 1
            if any(pn == n and abs(px - x) < max(w, pw) / 2 and abs(py - y) < max(h, ph) / 2 for pn, px, py, pw, ph in pred):
                found += 1
    return found / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=25)
    a = ap.parse_args()
    imgsz = json.loads((MODELS / "clashai_yolo11s.json").read_text())["imgsz"] if (MODELS / "clashai_yolo11s.json").exists() else 896
    own = sorted(p.stem for p in (DATA / "labels/train_fk").glob("own_*.txt"))
    test = [s for s in own if is_test_game(s)]
    train = [s for s in own if not is_test_game(s)]
    print(f"nos images : {len(train)} pour apprendre, {len(test)} pour tester", flush=True)
    # petit jeu d'entraînement : nos images x4 + échantillon des images habituelles
    shutil.rmtree(QUICK, ignore_errors=True)
    for sub in ("images/train", "labels/train"):
        (QUICK / sub).mkdir(parents=True)
    random.seed(0)
    others = random.sample(sorted((DATA / "images/train").glob("*.jpg")), 3000) + \
        random.sample(sorted((DATA / "images/train_fk").glob("fk_*.jpg")), 1500)
    for rep in range(4):
        for s in train:
            shutil.copy(DATA / "images/train_fk" / f"{s}.jpg", QUICK / "images/train" / f"r{rep}_{s}.jpg")
            shutil.copy(DATA / "labels/train_fk" / f"{s}.txt", QUICK / "labels/train" / f"r{rep}_{s}.txt")
    for img in others:
        lab = img.parent.parent.parent / "labels" / img.parent.name / f"{img.stem}.txt"
        shutil.copy(img, QUICK / "images/train" / img.name)
        if lab.exists():
            shutil.copy(lab, QUICK / "labels/train" / f"{img.stem}.txt")
    d = yaml.safe_load(open(DATA / "data_v2.yaml"))
    d.update(path=str(QUICK), train="images/train", val=str(DATA / "images/val"))
    yaml.safe_dump(d, open(QUICK / "data.yaml", "w"), sort_keys=False)

    before_own = own_recall(MODELS / "clashai_yolo11s.engine", imgsz, test)
    print(f"avant : nos unités vues {before_own:.0%}", flush=True)
    subprocess.run([str(PY), "scripts/train_yolo.py", "--model", str(MODELS / "clashai_yolo11s.pt"), "--data", str(QUICK / "data.yaml"),
                    "--hours", str(a.minutes / 60), "--lr0", "0.002", "--close-mosaic", "0", "--workers", "8",
                    "--name", "quick_ft", "--imgsz", str(imgsz), "--batch", "12"], cwd=ROOT)
    best = ROOT / "runs/detector/quick_ft/weights/best.pt"
    subprocess.run([str(PY), "-c", f"from ultralytics import YOLO; YOLO(r'{best}').export(format='engine', half=True, imgsz={imgsz}, device=0)"], cwd=ROOT)
    engine = best.with_suffix(".engine")
    after_own = own_recall(engine, imgsz, test)

    def f1(model):
        out = subprocess.run([str(PY), "scripts/eval_yolo.py", str(model), f"--imgsz={imgsz}"], cwd=ROOT,
                             capture_output=True, text=True, encoding="utf-8").stdout
        return json.loads(out[out.find("{"):])["f1"]
    f_before, f_after = f1(MODELS / "clashai_yolo11s.engine"), f1(engine)
    ok = after_own > before_own and f_after >= f_before - 0.005
    print(f"nos unités vues : {before_own:.0%} -> {after_own:.0%} | cartes connues F1 : {f_before:.3f} -> {f_after:.3f}")
    if ok:
        for ext in (".pt", ".engine"):
            shutil.copy(MODELS / f"clashai_yolo11s{ext}", MODELS / f"clashai_yolo11s_prev{ext}")
        shutil.copy(best, MODELS / "clashai_yolo11s.pt")
        shutil.copy(engine, MODELS / "clashai_yolo11s.engine")
    print("ADOPTÉ" if ok else "PAS ADOPTÉ (on garde le détecteur actuel)")


if __name__ == "__main__":
    main()
