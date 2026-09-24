"""Actions de combat vérifiées : on ne suppose jamais qu'un tap a marché, on le voit."""
from __future__ import annotations

import time

import numpy as np

from clashai import battle as B
from clashai.device import Device


def _card_patch(img: np.ndarray, slot: int) -> np.ndarray:
    x0, y0, x1, y1 = B.card_box(img, slot)
    return img[max(y0 - 30, 0):y1, x0:x1].astype(np.int16)


def select_card(d: Device, slot: int, timeout: float = 0.3) -> float | None:
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
    return None


def play_card(d: Device, slot: int, target: tuple[int, int]) -> bool:
    """Sélectionne la carte (vérifié) puis la pose en `target`. Faux si la carte n'a pas réagi."""
    if select_card(d, slot) is None:
        return False
    d.tap(*target, hold=0.0)
    return True
