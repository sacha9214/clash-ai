"""Boucle de combat du cerveau v0 : voir -> comprendre -> décider -> jouer (vérifié)."""
from __future__ import annotations

import json
import os
import time
import unicodedata

import cv2

from clashai import battle as B
from clashai import hand as H
from clashai.actions import play_card
from clashai.brain import Brain
from clashai.detect import Detector, draw


def _ascii(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


class Agent:
    def __init__(self, out: str = "runs/games"):
        self.det, self.brain, self.out = Detector(track=True), Brain(), out
        os.makedirs(out, exist_ok=True)

    def think(self, img, now, fps):
        units = self.det(img)
        h, w = img.shape[:2]
        seen = Brain.to_seen(units, w, h, self.det.trails, fps)
        hand = H.read_hand(img, min_score=0.45)
        ready = [B.card_ready(img, s) for s in range(4)]
        el = B.read_elixir(img)
        d = self.brain.decide(seen, hand, ready, el, now)
        info = [f"elixir {el:.1f}  main : " + ", ".join(c or "?" for c in hand)]
        return units, d, info, hand, el

    def annotate(self, img, units, d, info):
        v = img.copy()
        draw(v, units, self.det.trails)
        if d:
            h, w = v.shape[:2]
            p = (int(d.x * w), int(d.y * h))
            cv2.circle(v, p, 16, (0, 255, 255), 3)
            cv2.arrowedLine(v, B.px(v, B.CARD_X[d.slot], B.CARD_Y - 0.05), p, (0, 255, 255), 2)
            info = [d.reason] + info
        for i, line in enumerate(info):
            line = _ascii(line)
            cv2.putText(v, line, (8, 80 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(v, line, (8, 80 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return v

    def play_battle(self, dev, game_id: str, params: dict | None = None) -> dict:
        """Joue un combat jusqu'au bout avec la variante de stratégie `params`."""
        self.brain = Brain(params)
        folder = os.path.join(self.out, game_id)
        os.makedirs(folder, exist_ok=True)
        self.det.tracker.reset() if self.det.tracker is not None else None
        log, last_play, gone, t_prev, n, refused = [], 0.0, None, time.time(), 0, 0
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
            fps, t_prev = 1 / max(now - t_prev, 1e-3), now
            units, d, info, hand, el = self.think(img, now, fps)
            if d and now - last_play > 0.8:
                h, w = img.shape[:2]
                ok = play_card(dev, d.slot, (int(d.x * w), int(d.y * h)))
                log.append({"t": round(now, 2), "card": d.card, "x": round(d.x, 3), "y": round(d.y, 3),
                            "reason": d.reason, "ok": ok, "elixir": el, "hand": hand,
                            "units": [(u.name, u.enemy, u.center) for u in units]})
                if ok:
                    last_play = time.time()
                    cv2.imwrite(os.path.join(folder, f"play{n:03d}.jpg"), self.annotate(img, units, d, info))
                    n += 1
                else:
                    refused += 1
                    last_play = time.time() - 0.4
        with open(os.path.join(folder, "decisions.jsonl"), "w") as f:
            for row in log:
                f.write(json.dumps(row, default=str) + "\n")
        return {"game": game_id, "played": n, "refused": refused, "params": self.brain.p}
