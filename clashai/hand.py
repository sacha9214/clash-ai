"""Reconnaissance des 4 cartes en main par comparaison à des exemples.

Comparaison en niveaux de gris normalisés : marche aussi pour une carte grisée
(pas assez d'élixir). Exemples dans assets/card_templates.npz, extraits d'un vrai
combat (script : scripts/build_card_templates.py à venir pour d'autres decks).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from clashai import battle as B

_T = np.load(Path(__file__).resolve().parents[1] / "assets/card_templates.npz")
SIZE = (56, 72)


def _feat(crop: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(cv2.resize(crop, SIZE), cv2.COLOR_BGR2GRAY)[:52].astype(np.float32)
    g = (g - g.mean()) / (g.std() + 1e-6)
    return g.ravel() / np.sqrt(g.size)


_FEATS = np.stack([_feat(im) for im in _T["images"]])
_LABELS = _T["labels"]


def card_crop(img: np.ndarray, slot: int) -> np.ndarray:
    x0, y0, x1, y1 = B.card_box(img, slot)
    return img[y0 + 5:y1 - 5, x0 + 5:x1 - 5]


def identify(crop: np.ndarray) -> tuple[str, float]:
    """(nom de la carte, score de ressemblance 0-1)."""
    s = _FEATS @ _feat(crop)
    i = int(s.argmax())
    return str(_LABELS[i]), float(s[i])


def read_hand(img: np.ndarray, min_score: float = 0.55) -> list[str | None]:
    """Les 4 cartes en main (None si illisible ou emplacement vide)."""
    out = []
    for slot in range(4):
        name, score = identify(card_crop(img, slot))
        out.append(name if score >= min_score and name != "empty" else None)
    return out


def known_cards() -> set[str]:
    return set(str(x) for x in _LABELS)


def learn(crop: np.ndarray, name: str) -> None:
    """Ajoute un exemple (carte nouvellement mise dans le deck) et l'enregistre."""
    global _FEATS, _LABELS
    img = cv2.resize(crop, SIZE)
    _FEATS = np.vstack([_FEATS, _feat(img)[None]])
    _LABELS = np.append(_LABELS, name)
    path = Path(__file__).resolve().parents[1] / "assets/card_templates.npz"
    old = np.load(path)
    np.savez_compressed(path, images=np.concatenate([old["images"], img[None]]),
                        labels=np.append(old["labels"], name))
