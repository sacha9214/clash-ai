"""Fenêtre de visualisation en direct : l'écran du téléphone + ce que l'IA comprend.

Lancer : .venv/bin/python -m clashai.viewer   (q ou Échap pour quitter)
Les couches (détections, suivi, élixir…) s'ajoutent via `Overlay.layers`.
"""
from __future__ import annotations

import collections
import time

import av  # noqa: F401  (importé avant cv2 : voir README, doublon de libavdevice)
import cv2
import numpy as np

from clashai.device import Device

FONT = cv2.FONT_HERSHEY_SIMPLEX


class RateMeter:
    def __init__(self, window: float = 1.0):
        self.window, self.times = window, collections.deque()

    def tick(self, t: float) -> None:
        self.times.append(t)
        while self.times and t - self.times[0] > self.window:
            self.times.popleft()

    @property
    def rate(self) -> float:
        return len(self.times) / self.window


def hud(img: np.ndarray, lines: list[str]) -> None:
    pad, lh = 8, 22
    w = max(cv2.getTextSize(l, FONT, 0.55, 1)[0][0] for l in lines) + 2 * pad
    h = lh * len(lines) + pad
    roi = img[0:h, 0:w]
    roi[:] = (roi * 0.35).astype(np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(img, line, (pad, pad + 14 + i * lh), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def run(layers=()):
    with Device() as dev:
        stream, shown = RateMeter(), RateMeter()
        last_n = 0
        cv2.namedWindow("Clash AI", cv2.WINDOW_AUTOSIZE)
        while True:
            dev.wait_frame(timeout=0.5, after=last_n)
            img, t_recv, n = dev.frame()
            for i in range(last_n, n):
                stream.tick(time.perf_counter())
            last_n = n
            view = img.copy()
            t0 = time.perf_counter()
            infos = []
            for layer in layers:
                infos += layer(view) or []
            t_proc = (time.perf_counter() - t0) * 1000
            shown.tick(time.perf_counter())
            age = (time.perf_counter() - t_recv) * 1000
            hud(view, [f"flux {stream.rate:4.0f} img/s  affiche {shown.rate:4.0f}",
                       f"analyse {t_proc:5.1f} ms  age image {age:5.1f} ms", *infos])
            cv2.imshow("Clash AI", view)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27) or cv2.getWindowProperty("Clash AI", cv2.WND_PROP_VISIBLE) < 1:
                break
        cv2.destroyAllWindows()


def battle_layer(view: np.ndarray) -> list[str]:
    """Couche phase 1 : élixir lu à l'écran + cartes jouables encadrées."""
    from clashai import battle as B
    if not B.in_battle(view):
        return ["hors combat"]
    for s in range(4):
        x0, y0, x1, y1 = B.card_box(view, s)
        cv2.rectangle(view, (x0, y0), (x1, y1), (0, 220, 0) if B.card_ready(view, s) else (0, 0, 230), 2)
    return [f"elixir {B.read_elixir(view):.1f}"]


if __name__ == "__main__":
    run([battle_layer])
