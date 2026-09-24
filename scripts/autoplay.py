"""Enchaîne de vrais combats (Ladder) toute la nuit et tient un journal.

États reconnus : accueil (bouton Battle), recherche d'adversaire, combat, écran de fin (OK).
Tout écran inconnu qui dure -> capture + arrêt, pour ne jamais cliquer à l'aveugle
(boutique, offres payantes…).

  .venv-katacr/bin/python scripts/autoplay.py --games 10
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import av  # noqa: F401,E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from clashai import battle as B  # noqa: E402
from clashai.agent import Agent  # noqa: E402
from clashai.device import Device  # noqa: E402
from clashai import strategy as ST  # noqa: E402

BATTLE_BTN = (0.50, 0.822)       # bouton jaune « Battle » de l'accueil = même endroit que « OK » en fin de combat


def color_at(img, fx, fy, r=6):
    x, y = B.px(img, fx, fy)
    return img[y - r:y + r, x - r:x + r].reshape(-1, 3).mean(0)   # BGR


def is_home(img):
    b, g, r = color_at(img, 0.372, 0.797)       # coin du bouton Battle (hors texte)
    return r > 120 and g > 80 and b < 70 and r > g   # jaune (même assombri par un tutoriel)


def is_blue_btn(img, fx, fy):
    b, g, r = color_at(img, fx, fy)
    return b > 200 and g > 120 and r < 120


def end_ok(img):
    """Position du bouton OK bleu de fin de combat (seul au centre, ou à droite de « Play Again »)."""
    for fx, fy in ((0.40, 0.828), (0.60, 0.828)):
        if is_blue_btn(img, fx, fy):
            return (0.5 if fx == 0.40 else 0.655), 0.825
    return None


def is_end(img):
    return end_ok(img) is not None


def is_chest(img):
    """Écran d'ouverture de coffre : fond uni et coloré (orange, bleu, violet…) tout
    en haut, sans l'interface de l'accueil ni l'herbe de l'arène."""
    pts = np.array([color_at(img, fx, 0.04, r=8) for fx in (0.08, 0.5, 0.92)])
    uniform = np.abs(pts - pts.mean(0)).max() < 25
    sat = (pts.max(1) - pts.min(1)).mean()
    return bool(uniform and sat > 60)


def result_of(img):
    """Compare les couronnes dorées : les nôtres (bandeau bleu, bas) et les siennes (bandeau rouge, haut)."""
    h, w = img.shape[:2]

    def gold(y0, y1):
        band = img[int(y0 * h):int(y1 * h), int(0.12 * w):int(0.88 * w)].reshape(-1, 3).astype(int)
        b, g, r = band[:, 0], band[:, 1], band[:, 2]
        return ((r > 200) & (g > 140) & (b < 90) & (r - b > 130)).mean()

    ours, theirs = gold(0.44, 0.55), gold(0.17, 0.32)
    if max(ours, theirs) < 0.01:
        return "draw"
    return "win" if ours > theirs else "loss"


ap = argparse.ArgumentParser()
ap.add_argument("--games", type=int, default=5)
ap.add_argument("--out", default="runs/games")
a = ap.parse_args()
agent = Agent(a.out)
journal = os.path.join(a.out, "journal.jsonl")

with Device() as dev:
    played, unknown_since, chest_until = 0, None, 0.0
    while played < a.games:
        img, _, _ = dev.frame()
        if B.in_battle(img):
            gid = time.strftime("%Y%m%d-%H%M%S")
            print(f"[{gid}] combat {played + 1}/{a.games}", flush=True)
            stats = ST.load()
            params = ST.choose(stats)
            summary = agent.play_battle(dev, gid, params)
            time.sleep(2.5)
            img, _, _ = dev.frame()
            summary["result"] = result_of(img)
            cv2.imwrite(os.path.join(a.out, gid, "end.jpg"), img)
            ST.record(stats, params, summary["result"])
            with open(journal, "a") as f:
                f.write(json.dumps(summary) + "\n")
            print("   ", summary, flush=True)
            played += 1
            unknown_since = None
        elif is_end(img):
            dev.tap(*B.px(img, *end_ok(img)))
            time.sleep(3)
            # Juste après un combat : écrans d'ouverture de coffre / récompenses.
            # On tape au centre (seulement dans ce contexte, borné) jusqu'au retour à l'accueil.
            for _ in range(25):
                img, _, _ = dev.frame()
                if is_home(img) or B.in_battle(img):
                    break
                dev.tap(*B.px(img, 0.5, 0.42))
                time.sleep(1.3)
        elif is_home(img):
            if played >= a.games or os.path.exists(os.path.join(a.out, "STOP")):
                break
            dev.tap(*B.px(img, *BATTLE_BTN))
            time.sleep(4)
            unknown_since = None
        elif is_chest(img) or (chest_until and time.time() < chest_until):
            if is_chest(img):
                chest_until = time.time() + 8
            # alterne : centre (coffre, carte) et bouton OK des écrans « niveau supérieur »
            taps = getattr(dev, "_reward_taps", 0)
            dev._reward_taps = taps + 1
            dev.tap(*B.px(img, 0.5, 0.42 if taps % 2 == 0 else 0.66))
            time.sleep(1.3)
            unknown_since = None
        else:
            unknown_since = unknown_since or time.time()
            # Route des trophées / nouvelle arène : bouton OK tout en bas. Sans risque ailleurs
            # (sur l'accueil c'est la barre d'onglets). On l'essaie toutes les 10 s.
            waited = time.time() - unknown_since
            if waited > 10 and int(waited) % 10 == 0:
                dev.tap(*B.px(img, 0.5, 0.966))
                time.sleep(1.0)
            if time.time() - unknown_since > 45:   # matchmaking dure rarement plus
                cv2.imwrite(os.path.join(a.out, "unknown.jpg"), img)
                print("écran inconnu depuis 45 s : arrêt (voir unknown.jpg)", flush=True)
                break
            time.sleep(0.5)
print("fin")
