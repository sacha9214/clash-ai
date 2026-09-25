"""Actions de combat vérifiées : on ne suppose jamais qu'un tap a marché, on le voit."""
from __future__ import annotations

import time
from typing import Callable

import numpy as np

from clashai import battle as B
from clashai.device import Device


def _card_patch(img: np.ndarray, slot: int) -> np.ndarray:
    x0, y0, x1, y1 = B.card_box(img, slot)
    return img[max(y0 - 30, 0):y1, x0:x1].astype(np.int16)


def select_card(d: Device, slot: int, timeout: float = 0.3, on_frame: Callable | None = None) -> float | None:
    """Touche la carte et attend qu'elle se soulève. Renvoie la latence (ms) ou None."""
    img, _, n = d.frame()
    ref = _card_patch(img, slot)
    t0 = time.perf_counter()
    d.tap(*B.px(img, B.CARD_X[slot], B.CARD_Y), hold=0.0)
    while time.perf_counter() - t0 < timeout:
        d.wait_frame(timeout=0.1, after=n)
        img, t_recv, n = d.frame()
        if np.abs(_card_patch(img, slot) - ref).mean() > 12:
            return (t_recv - t0) * 1000
        if on_frame:
            on_frame(img)
    return None


def play_card(d: Device, slot: int, target: tuple[int, int], timeout: float = 0.7,
              on_frame: Callable | None = None) -> bool:
    """Sélectionne la carte (vérifié), la pose en `target`, puis vérifie qu'elle est
    vraiment partie. Si le jeu refuse (endroit interdit, élixir), la carte reste
    soulevée : on la désélectionne pour ne pas laisser le jeu dans un état bancal.
    on_frame(img) est appelé sur les images reçues pendant l'attente : l'appelant continue
    de regarder le combat (fenêtre, suivi des unités) au lieu d'être aveugle."""
    if select_card(d, slot, on_frame=on_frame) is None:
        return False
    time.sleep(0.05)                       # fin de l'animation de soulèvement
    img, _, n = d.frame()
    lifted = _card_patch(img, slot)
    d.tap(*target, hold=0.0)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        d.wait_frame(timeout=0.1, after=n)
        img, _, n = d.frame()
        if np.abs(_card_patch(img, slot) - lifted).mean() > 15:   # la carte a quitté la main
            return True
        if on_frame:
            on_frame(img)
    d.tap(*B.px(img, B.CARD_X[slot], B.CARD_Y), hold=0.0)   # refusée : désélectionner
    return False
