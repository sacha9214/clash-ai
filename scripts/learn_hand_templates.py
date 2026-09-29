"""Apprend les images de la main (cartes en couleur ET grisées/en charge) depuis nos matchs, sans étiquetage humain.

Une carte reste dans son emplacement jusqu'à ce qu'on la joue : entre la pose précédente depuis cet emplacement
(+1 s, le temps que la suivante arrive) et le moment où l'IA joue la carte X depuis lui, toutes les captures de cet
emplacement montrent X. On garde jusqu'à --per exemples par carte (moitié en couleur, moitié grisés), ajoutés à
assets/card_templates.npz.

  .venv-yolo\\Scripts\\python scripts/learn_hand_templates.py --since 20260927-18 --per 40
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="20260927-18")
    ap.add_argument("--per", type=int, default=40)
    a = ap.parse_args()
    from clashai import battle as B
    from clashai import hand as H
    from clashai.cards import DECK
    found = collections.defaultdict(lambda: {"color": [], "grey": []})
    for game in sorted(glob.glob(str(ROOT / "runs/games/2026*"))):
        g = Path(game).name
        if g < a.since or not (Path(game) / "decisions.jsonl").exists():
            continue
        plays = [json.loads(l) for l in open(Path(game) / "decisions.jsonl", encoding="utf-8")]
        plays = [p for p in plays if p.get("ok") and p["card"] in DECK]
        caps = {int(Path(f).stem): f for f in glob.glob(str(ROOT / "runs/capture" / g / "*.jpg"))}
        if not caps:
            continue
        # emplacement de chaque pose : la carte jouée était dans la main à la décision, à cet emplacement
        last_by_slot: dict[int, float] = {}
        for p in plays:
            hand = p.get("hand") or []
            if p["card"] not in hand:
                continue
            slot = p["slot"] if "slot" in p else hand.index(p["card"])
            # fenêtre courte (3 s avant la pose) : une pose précédente depuis cet emplacement peut manquer au journal
            # (carte illisible à la décision) -> une longue fenêtre mélangeait l'ancienne carte (29/09 : Mousquetaire
            # évoluée apprise comme « Canon »)
            t0 = max(last_by_slot.get(slot, p["t"] - 12) + 1.2, p["t"] - 3.0)
            last_by_slot[slot] = p["t"]
            k0, k1 = int(t0 * 10) % 10**7, int((p["t"] - 0.2) * 10) % 10**7
            for k, f in caps.items():
                if k0 <= k <= k1:
                    img = cv2.imread(f)
                    if img is None or not B.in_battle(img):
                        continue
                    crop = H.card_crop(img, slot)
                    name, score = H.identify(crop)       # anciens exemples : s'ils disent nettement AUTRE CHOSE, on jette
                    if name != p["card"] and score > 0.45:
                        continue                         # le modèle actuel y voit autre chose : on n'apprend pas un doute
                    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                    kind = "color" if hsv[..., 1].mean() > 60 else "grey"
                    found[p["card"]][kind].append(cv2.resize(crop, H.SIZE))
    path = ROOT / "assets/card_templates.npz"
    old = np.load(path)
    imgs, labels = list(old["images"]), list(old["labels"])
    have = {hashlib.md5(im.tobytes()).hexdigest() for im in imgs}   # relancer le script ne rajoute pas les mêmes images
    random.seed(0)
    for card, kinds in found.items():
        add = []
        for kind, lst in kinds.items():
            new = [im for im in lst if hashlib.md5(im.tobytes()).hexdigest() not in have]
            add += random.sample(new, min(len(new), a.per // 2))
        have |= {hashlib.md5(im.tobytes()).hexdigest() for im in add}
        imgs += add
        labels += [card] * len(add)
        print(f"{card:12} +{len(add)} ({len(kinds['color'])} en couleur, {len(kinds['grey'])} grisés trouvés)")
    # écrit à côté puis remplace : un arrêt en pleine écriture ne laisse pas un fichier tronqué (hand.py le charge à
    # l'import : tous les scripts planteraient)
    tmp = path.with_name("card_templates.tmp.npz")
    np.savez_compressed(tmp, images=np.stack(imgs), labels=np.array(labels))
    os.replace(tmp, path)
    print(f"-> {len(labels)} exemples dans {path.name}")


if __name__ == "__main__":
    main()
