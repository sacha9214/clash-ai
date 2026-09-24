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


def run(layers=(), detect: bool = False, record: str | None = None, seconds: float | None = None):
    import signal
    # `kill` = sortie propre (la vidéo enregistrée doit être finalisée)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt))
    writer = None
    t_start, last_write, last_n = time.time(), 0.0, 0
    stream, shown = RateMeter(), RateMeter()
    try:
        with Device() as dev:
            layers = list(layers)
            if detect:
                layers.insert(0, DetectionLayer(dev.frame))
            cv2.namedWindow("Clash AI", cv2.WINDOW_AUTOSIZE)
            while not (seconds and time.time() - t_start > seconds):
                dev.wait_frame(timeout=0.5, after=last_n)
                img, t_recv, n = dev.frame()
                for _ in range(last_n, n):
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
                if record and time.time() - last_write >= 1 / 30:   # vidéo à 30 img/s réelles
                    if writer is None:
                        writer = cv2.VideoWriter(record, cv2.VideoWriter_fourcc(*"avc1"), 30,
                                                 (view.shape[1], view.shape[0]))
                    writer.write(view)
                    last_write = time.time()
                k = cv2.waitKey(1) & 0xFF
                if k in (ord("q"), 27) or cv2.getWindowProperty("Clash AI", cv2.WND_PROP_VISIBLE) < 1:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        if writer is not None:
            writer.release()
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


class DetectionLayer:
    """Détection + suivi dans un thread à part : l'affichage reste à 60 img/s,
    les boîtes sont celles de la dernière image analysée (âge affiché)."""

    def __init__(self, dev_getter):
        import threading
        from clashai import battle as B
        from clashai.detect import Detector
        self.B, self.det = B, Detector(track=True)
        self.get_frame = dev_getter
        self.units, self.dt, self.age_t, self.rate = [], 0.0, 0.0, RateMeter(2.0)
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        last = -1
        while True:
            img, t_recv, n = self.get_frame()
            if img is None or n == last or not self.B.in_battle(img):
                time.sleep(0.005)
                continue
            last = n
            t0 = time.perf_counter()
            units = self.det(img)
            self.units, self.trails, self.age_t = units, {k: list(v) for k, v in self.det.trails.items()}, t_recv
            self.dt = (time.perf_counter() - t0) * 1000
            self.rate.tick(time.perf_counter())

    def __call__(self, view):
        from clashai.detect import draw
        if not self.units:
            return ["detection : en attente d'un combat"]
        draw(view, self.units, getattr(self, "trails", None))
        mine = sum(not u.enemy for u in self.units)
        return [f"detection {self.dt:4.0f} ms  {self.rate.rate:4.1f} img/s  retard {(time.perf_counter() - self.age_t) * 1000:4.0f} ms",
                f"unites : {mine} a nous, {len(self.units) - mine} ennemies"]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Fenêtre en direct de Clash AI")
    ap.add_argument("--detect", action="store_true", help="détection + suivi des unités (lancer avec .venv-katacr)")
    ap.add_argument("--record", help="enregistre la vue annotée dans ce fichier .mp4")
    ap.add_argument("--seconds", type=float, help="s'arrête tout seul après N secondes")
    a = ap.parse_args()
    run([battle_layer], detect=a.detect, record=a.record, seconds=a.seconds)
