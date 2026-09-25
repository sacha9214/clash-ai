"""Phase 2b : entraîne NOTRE détecteur, petit et rapide (un seul modèle, 150 classes).

Pipeline d'entraînement de KataCR (MIT) : chaque époque génère de nouvelles images
synthétiques en collant des sprites découpés sur des arènes, déjà annotées.
Validation sur leurs 6 939 images réelles annotées à la main.

  .venv-katacr/bin/python scripts/train_detector.py --model yolov8s --epochs 40
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KATACR = ROOT / "third_party/KataCR"
sys.path.insert(0, str(KATACR))
os.chdir(KATACR)   # leurs chemins de config sont relatifs à la racine KataCR
# ultralytics 8.1 charge ses .pt par pickle complet ; torch >= 2.6 (requis pour les RTX 50xx) le refuse
# par défaut. Modèles KataCR de confiance -> on garde l'ancien comportement.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import torch  # noqa: E402
import yaml  # noqa: E402
from ultralytics.cfg import get_cfg  # noqa: E402
from katacr.yolov8.train import YOLO_CR  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="yolov8s")
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--workers", type=int, default=6)
ap.add_argument("--name", default=None)
ap.add_argument("--hours", type=float, default=None, help="durée max d'entraînement")
ap.add_argument("--fraction", type=float, default=1.0, help="part des images de validation (tests rapides)")
a = ap.parse_args()

device = 0 if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
# data.yaml de KataCR avec un chemin absolu : on en écrit une copie qui pointe vers le dataset local
data = yaml.safe_load(open(KATACR / "katacr/yolov8/detector1/data.yaml", encoding="utf-8"))
data["path"] = str(ROOT / "data/Clash-Royale-Dataset/images/part2")
data_yaml = ROOT / "runs/detector/data.yaml"
data_yaml.parent.mkdir(parents=True, exist_ok=True)
yaml.safe_dump(data, open(data_yaml, "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)

cfg = dict(get_cfg("./katacr/yolov8/ClashRoyale.yaml"))
cfg.update(
    model=f"{a.model}.yaml", data=str(data_yaml),
    epochs=a.epochs, batch=a.batch, workers=a.workers, device=device, deterministic=False,
    project=str(ROOT / "runs/detector"), name=a.name or f"{a.model}_single", exist_ok=True,
    fraction=a.fraction, patience=8, time=a.hours,
)
YOLO_CR(f"{a.model}.yaml", task="detect").train(**cfg)
