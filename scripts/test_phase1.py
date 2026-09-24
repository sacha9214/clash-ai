"""Test de la phase 1 en Training Camp : latence réelle + l'IA joue (au hasard).

1. Latence : on touche une carte et on chronomètre jusqu'à l'image où elle se soulève
   (inclut : envoi du tap, réaction du jeu, encodage, USB, décodage).
2. Jeu : tant que le combat dure, pose une carte jouable sur notre moitié.
"""
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import av  # noqa: F401,E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from clashai import battle as B  # noqa: E402
from clashai.device import Device  # noqa: E402
from clashai.actions import play_card  # noqa: E402

OUT = next((a for a in sys.argv[1:] if not a.startswith("--")), "runs/phase1")
os.makedirs(OUT, exist_ok=True)


def patch(img, slot):
    x0, y0, x1, y1 = B.card_box(img, slot)
    return img[y0 - 30:y1, x0:x1].astype(np.int16)


def measure_latency(d: Device, slot: int) -> float | None:
    img, _, n = d.frame()
    ref = patch(img, slot)
    x, y = B.px(img, B.CARD_X[slot], B.CARD_Y)
    t0 = time.perf_counter()
    d.tap(x, y, hold=0.0)
    while time.perf_counter() - t0 < 1.0:
        d.wait_frame(timeout=0.2, after=n)
        img, t_recv, n = d.frame()
        if np.abs(patch(img, slot) - ref).mean() > 12:
            lat = (t_recv - t0) * 1000
            time.sleep(0.2)
            d.tap(x, y, hold=0.0)       # re-toucher la carte = la désélectionner
            time.sleep(0.4)
            return lat
    return None


def annotate(img, elixir, ready, target=None, slot=None):
    v = img.copy()
    for s in range(4):
        x0, y0, x1, y1 = B.card_box(v, s)
        cv2.rectangle(v, (x0, y0), (x1, y1), (0, 255, 0) if ready[s] else (0, 0, 255), 2)
    cv2.putText(v, f"elixir {elixir:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    if target:
        cv2.circle(v, target, 14, (0, 255, 255), 3)
        sx, sy = B.px(v, B.CARD_X[slot], B.CARD_Y)
        cv2.arrowedLine(v, (sx, sy - 60), target, (0, 255, 255), 2)
    return v


with Device() as d:
    img, _, _ = d.frame()
    if not B.in_battle(img):
        sys.exit("Pas en combat : lance un Training Camp d'abord.")

    lats = [] if '--no-latency' in sys.argv else [l for l in (measure_latency(d, s % 4) for s in range(6)) if l]
    if lats:
        print(f"latence tap->image : médiane {statistics.median(lats):.0f} ms, "
              f"min {min(lats):.0f}, max {max(lats):.0f}  ({len(lats)}/6 mesures)")

    plays, gone = 0, None
    want = random.randint(4, 7)
    while True:
        img, _, _ = d.frame()
        if not B.in_battle(img):
            gone = gone or time.time()
            if time.time() - gone > 4:
                break
            time.sleep(0.2)
            continue
        gone = None
        el = B.read_elixir(img)
        ready = [B.card_ready(img, s) for s in range(4)]
        if el >= want and any(ready):
            slot = random.choice([s for s in range(4) if ready[s]])
            (ax, bx), (ay, by) = B.OWN_HALF
            target = B.px(img, random.uniform(ax, bx), random.uniform(ay, by))
            cv2.imwrite(f"{OUT}/play{plays:02d}.png", annotate(img, el, ready, target, slot))
            if not play_card(d, slot, target):
                continue            # carte absente ou pas prête : on réévalue l'image suivante
            plays += 1
            want = random.randint(4, 7)
            time.sleep(0.6)
        time.sleep(0.05)
    img, _, _ = d.frame()
    cv2.imwrite(f"{OUT}/end.png", img)
    print(f"combat terminé, {plays} cartes jouées")
