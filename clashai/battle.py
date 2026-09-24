"""Lecture de l'écran de combat (élixir, main) et positions utiles.

Toutes les positions sont en fractions de l'image (0-1) pour ne pas dépendre
de la résolution du flux. Mesurées sur le REDMAGIC 11 Air (ratio 1216x2688).
"""
from __future__ import annotations

import numpy as np

# Centres des 4 cartes en main, et rangée de la barre d'élixir
CARD_X = (0.311, 0.498, 0.687, 0.875)
CARD_Y = 0.910
CARD_HALF = (0.08, 0.045)           # demi-largeur / demi-hauteur d'une carte
ELIXIR_Y = 0.981
ELIXIR_X = (0.285, 0.955)           # début / fin de la partie graduée de la barre
# Zone où l'on peut poser une troupe (notre moitié, hors tours du bas)
OWN_HALF = ((0.08, 0.92), (0.46, 0.66))


def px(img: np.ndarray, fx: float, fy: float) -> tuple[int, int]:
    h, w = img.shape[:2]
    return int(fx * w), int(fy * h)


def card_box(img: np.ndarray, slot: int) -> tuple[int, int, int, int]:
    h, w = img.shape[:2]
    cx, cy = CARD_X[slot] * w, CARD_Y * h
    dx, dy = CARD_HALF[0] * w, CARD_HALF[1] * h
    return int(cx - dx), int(cy - dy), int(cx + dx), int(cy + dy)


def in_battle(img: np.ndarray) -> bool:
    """Vrai si la barre d'élixir est visible en bas, quel que soit son remplissage.

    Partie pleine = rose (B et R élevés), partie vide = bleu foncé (B moyen, R≈0) :
    dans les deux cas le vert est bas et le bleu dominant.
    """
    h, w = img.shape[:2]
    y = int(ELIXIR_Y * h)
    row = img[y - 1:y + 2, int(ELIXIR_X[0] * w):int(ELIXIR_X[1] * w)].reshape(-1, 3).astype(int)
    b, g = row[:, 0], row[:, 1]
    return ((b > 90) & (g < 100)).mean() > 0.8


def read_elixir(img: np.ndarray) -> float:
    """Élixir 0-10 : proportion remplie (rose vif) de la barre graduée."""
    h, w = img.shape[:2]
    y = int(ELIXIR_Y * h)
    x0, x1 = int(ELIXIR_X[0] * w), int(ELIXIR_X[1] * w)
    row = img[y - 2:y + 3, x0:x1].astype(int).mean(axis=0)
    b, g, r = row[:, 0], row[:, 1], row[:, 2]
    filled = r > 150          # vide = bleu foncé (R≈0), plein = rose (R>200)
    return round(10 * filled.mean(), 1)


def card_ready(img: np.ndarray, slot: int) -> bool:
    """Carte jouable = colorée (une carte trop chère est grisée) et présente
    (un emplacement vide pendant la pioche est un aplat bleu, peu contrasté)."""
    x0, y0, x1, y1 = card_box(img, slot)
    patch = img[y0:y1, x0:x1].astype(int)
    sat = patch.max(axis=2) - patch.min(axis=2)
    return sat.mean() > 45 and patch.std() > 35
