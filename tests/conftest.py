"""Tests hors téléphone : le dépôt est importable depuis tests/ (pytest -q tests)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("YOLO_VERBOSE", "False")
