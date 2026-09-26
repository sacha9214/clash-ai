"""Fait jouer le cerveau v0 (règles tactiques) en direct, ou à blanc sur une vidéo.

  .venv-yolo\Scripts\python scripts/play_smart.py                 # joue sur le téléphone
  .venv-yolo\Scripts\python scripts/play_smart.py --dry video.mp4 # décisions seules, sur un enregistrement
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import av  # noqa: E402
import cv2  # noqa: E402

from clashai import battle as B  # noqa: E402
from clashai import hand as H  # noqa: E402
from clashai.brain import Brain  # noqa: E402
from clashai.detect import Detector, draw  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dry", help="vidéo enregistrée : afficher les décisions sans jouer")
ap.add_argument("--out", default="runs/smart")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

det, brain = Detector(track=True), Brain()


def annotate(img, units, d, info):
    v = img.copy()
    draw(v, units, det.trails)
    if d:
        h, w = v.shape[:2]
        p = (int(d.x * w), int(d.y * h))
        cv2.circle(v, p, 16, (0, 255, 255), 3)
        cv2.arrowedLine(v, B.px(v, B.CARD_X[d.slot], B.CARD_Y - 0.05), p, (0, 255, 255), 2)
        info = [d.reason] + info
    import unicodedata
    info = [unicodedata.normalize("NFKD", l).encode("ascii", "ignore").decode() for l in info]
    for i, line in enumerate(info):
        cv2.putText(v, line, (8, 80 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(v, line, (8, 80 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return v


def think(img, now, fps):
    units = det(img)
    h, w = img.shape[:2]
    seen = Brain.to_seen(units, w, h, det.trails, fps)
    hand = H.read_hand(img, min_score=0.45)
    ready = [B.card_ready(img, s) for s in range(4)]
    el = B.read_elixir(img)
    d = brain.decide(seen, hand, ready, el, now)
    info = [f"elixir {el:.1f}  main : " + ", ".join(c or "?" for c in hand)]
    return units, d, info


n_dec = 0
if a.dry:
    c = av.open(a.dry)
    fps = float(c.streams.video[0].average_rate)
    step = int(fps * 0.25)
    last_play = -9
    for i, f in enumerate(c.decode(video=0)):
        if i % step:
            continue
        img, t = f.to_ndarray(format="bgr24"), i / fps
        if not B.in_battle(img):
            continue
        units, d, info = think(img, t, 1 / 0.25)
        if d and t - last_play > 1.0:
            last_play = t
            print(f"{t:6.1f}s  {d.reason:45}  ({d.x:.2f}, {d.y:.2f})")
            cv2.imwrite(f"{a.out}/dry{n_dec:03d}.png", annotate(img, units, d, info))
            n_dec += 1
else:
    from clashai.actions import play_card
    from clashai.device import Device
    with Device() as dev:
        last_play, gone, t_prev = 0.0, None, time.time()
        while True:
            img, _, _ = dev.frame()
            if not B.in_battle(img):
                gone = gone or time.time()
                if time.time() - gone > 5:
                    break
                time.sleep(0.2)
                continue
            gone = None
            now = time.time()
            fps = 1 / max(now - t_prev, 1e-3)
            t_prev = now
            units, d, info = think(img, now, fps)
            if d and now - last_play > 0.8:
                h, w = img.shape[:2]
                ok = play_card(dev, d.slot, (int(d.x * w), int(d.y * h)))
                if not ok:
                    print(f"  refusé par le jeu : {d.reason}", flush=True)
                    last_play = time.time() - 0.4
                if ok:
                    last_play = time.time()
                    print(f"{d.reason:45} ({d.x:.2f}, {d.y:.2f})", flush=True)
                    cv2.imwrite(f"{a.out}/play{n_dec:03d}.png", annotate(img, units, d, info))
                    n_dec += 1
    print(f"combat terminé, {n_dec} cartes jouées")
