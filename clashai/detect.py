"""Choix du détecteur d'unités (même interface pour le reste de l'IA : Unit, Detector, draw, ARENA…).

- Par défaut : notre YOLO11s TensorRT (detect_yolo.py, .venv-yolo) — plus précis et 3x plus rapide que KataCR.
- CLASHAI_DETECTOR=katacr, ou si l'ancien ultralytics 8.1 est installé (.venv-katacr) : les détecteurs KataCR
  (detect_katacr.py), gardés en secours et pour les outils qui en dépendent (extraction vidéo, pré-étiquetage).
"""
import os

import ultralytics

_old = tuple(int(x) for x in ultralytics.__version__.split(".")[:2]) < (8, 2)
if os.environ.get("CLASHAI_DETECTOR") == "katacr" or _old:
    from clashai.detect_katacr import *  # noqa: F401,F403
    from clashai.detect_katacr import UI  # noqa: F401
else:
    from clashai.detect_yolo import *  # noqa: F401,F403
    from clashai.detect_yolo import UI  # noqa: F401
