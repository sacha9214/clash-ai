"""Étiquettes justes pour NOS unités, sans travail humain : l'IA sait où et quand elle a posé chaque carte.

Dans la seconde qui suit une pose (déploiement ~1 s), nos unités sont à l'endroit du tap : on les étiquette là,
même si le détecteur ne les voit pas (ex. nos Archères, presque jamais détectées sur notre écran).
Les autres unités de l'image viennent du détecteur (boîtes sûres, conf >= 0.7) ; les images où une de nos unités
posée récemment n'a pas pu être placée sont écartées (sinon on apprendrait « rien ici »).

Écrit D:/clash-ai-dataset/images/train_fk/own_*.jpg (+ étiquettes), classes de data_v2.yaml.

  .venv-yolo\\Scripts\\python scripts/own_labels.py
"""
from __future__ import annotations

import glob
import json
import statistics as st
import sys
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("D:/clash-ai-dataset")
sys.path.insert(0, str(ROOT))
# unités produites par nos cartes (nombre, écart horizontal en fraction d'arène entre elles)
OUR_UNITS = {"archers": ("archer", 2, 0.035), "musketeer": ("musketeer", 1, 0), "minions": ("minion", 3, 0.03), "knight": ("knight", 1, 0), "valkyrie": ("valkyrie", 1, 0),
             "mini-pekka": ("mini-pekka", 1, 0), "giant": ("giant", 1, 0), "cannon": ("cannon", 1, 0)}
WINDOW = (1.05, 1.7)         # juste après le déploiement (~1 s) : unité apparue, pas encore partie


def box_sizes(names_v2: dict) -> dict:
    """Taille médiane (w, h en fraction de l'arène) de chaque unité, mesurée sur les images réelles annotées."""
    sizes = {}
    for f in list((DATA / "labels/val").glob("*.txt"))[:600]:
        for l in f.read_text().splitlines():
            c, x, y, w, h = l.split()[:5]
            sizes.setdefault(names_v2[int(c)].rsplit("_", 1)[0], []).append((float(w), float(h)))
    return {k: (st.median(w for w, _ in v), st.median(h for _, h in v)) for k, v in sizes.items()}


def main():
    from clashai.detect_yolo import ARENA, Detector
    names_v2 = yaml.safe_load(open(DATA / "data_v2.yaml"))["names"]
    cid = {n: i for i, n in names_v2.items()}
    sizes = box_sizes(names_v2)
    det = Detector(track=False, conf=0.7)
    (DATA / "images/train_fk").mkdir(parents=True, exist_ok=True)
    n_img = n_own = 0
    for game in sorted(glob.glob(str(ROOT / "runs/games/2026*"))):
        dec_f, cap_dir = Path(game) / "decisions.jsonl", ROOT / "runs/capture" / Path(game).name
        if not dec_f.exists() or not cap_dir.exists():
            continue
        plays = [json.loads(l) for l in open(dec_f, encoding="utf-8")]
        plays = [p for p in plays if p.get("ok") and p["card"] in OUR_UNITS]
        caps = {int(Path(f).stem): f for f in glob.glob(str(cap_dir / "*.jpg"))}
        for key, f in caps.items():
            recent = []
            for p in plays:
                dt = (key - int(p["t"] * 10) % 10**7) / 10
                if WINDOW[0] <= dt <= WINDOW[1]:
                    recent.append(p)
            if not recent:
                continue
            img = cv2.imread(f)
            crop, _ = det._crop(img)
            r = det.model.predict(crop, imgsz=det.imgsz, conf=0.7, verbose=False, device=0)[0]
            lines = []
            own_boxes = []
            for p in recent:
                unit, count, spread = OUR_UNITS[p["card"]]
                w, h = sizes.get(unit, (0.08, 0.06))
                # tap (fractions de l'écran) -> fractions de l'arène recadrée ; le tap est aux pieds de l'unité
                cx = (p["x"] - ARENA[0]) / ARENA[2]
                cy = (p["y"] - ARENA[1]) / ARENA[3] - h / 2
                for i in range(count):
                    x = cx + (i - (count - 1) / 2) * spread
                    own_boxes.append((x, cy, w, h))
                    lines.append(f"{cid[unit + '_0']} {x:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            for (x, y, w, h), c in zip(r.boxes.xywhn.tolist(), r.boxes.cls.int().tolist()):
                name = r.names[c]
                if name.endswith("_0") and any(abs(x - ox) < ow and abs(y - oy) < oh for ox, oy, ow, oh in own_boxes):
                    continue                        # déjà étiquetée par nous (évite les doublons)
                if name in cid:
                    lines.append(f"{cid[name]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
            stem = f"own_{Path(game).name}_{key}"
            cv2.imwrite(str(DATA / "images/train_fk" / f"{stem}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            (DATA / "labels/train_fk" / f"{stem}.txt").write_text("\n".join(lines))
            n_img += 1
            n_own += len(own_boxes)
    print(f"{n_img} images de nos matchs avec {n_own} unités à nous étiquetées d'après nos poses")


if __name__ == "__main__":
    main()
