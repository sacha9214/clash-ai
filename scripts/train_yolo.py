"""Entraîne un détecteur récent (YOLO11, ultralytics à jour) sur les images de gen_synthetic.py.

Un seul modèle, 300 classes (unité x camp). Validation sur les images réelles de KataCR.
Tourne dans .venv-yolo (ultralytics 8.4 + torch cu128) :

  .venv-yolo\\Scripts\\python scripts/train_yolo.py --model yolo11s.pt --hours 8
"""
import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11s.pt", help="poids pré-entraînés (COCO) de départ")
    ap.add_argument("--data", default="D:/clash-ai-dataset/data.yaml")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--hours", type=float, default=None, help="durée max (prioritaire sur --epochs)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=896)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    if sys.platform == "win32":
        # empêche la mise en veille pendant l'entraînement (comme un lecteur vidéo) ; relâché à la fin du programme,
        # aucun réglage Windows n'est modifié
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    YOLO(a.model).train(
        data=a.data, epochs=a.epochs, time=a.hours, batch=a.batch, imgsz=a.imgsz, workers=a.workers,
        device=0, project=str(ROOT / "runs/detector"), name=a.name or Path(a.model).stem + "_cr", exist_ok=True,
        patience=15, cos_lr=True, close_mosaic=10, cache=False,
        # le camp se lit à la couleur (bleu = nous, rouge = eux) : pas de changement de teinte
        hsv_h=0.0, hsv_s=0.3, hsv_v=0.3,
        fliplr=0.5, flipud=0.0, degrees=0.0, mosaic=0.5, mixup=0.0,
    )


# Windows : les processus de chargement réimportent ce script
if __name__ == "__main__":
    main()
