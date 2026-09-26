"""Fabrique des images d'entraînement pour les cartes manquantes (nouvelles cartes, héros, évolutions) à partir
des rendus du Fan Kit officiel : on les colle, à la taille du jeu, dans nos images synthétiques existantes.

Écart à combler : les rendus sont vus de face et en grand ; en jeu, les unités sont petites et vues d'en haut.
D'où : taille 45-140 px, léger écrasement vertical, flou, variations de lumière, miroir. Le camp (bleu = nous,
rouge = eux) ne se voit pas sur un rendu : on dessine la petite barre de vie colorée au-dessus, comme en jeu.
Quelques images réelles étiquetées (label_tool.py) servent ensuite d'ancrage.

Écrit D:/clash-ai-dataset/images/train_fk/fk_*.jpg (+ labels/train_fk) et D:/clash-ai-dataset/data_v2.yaml
(les 300 classes actuelles, puis les nouvelles à la suite : les numéros existants ne bougent pas).

  .venv-yolo\\Scripts\\python scripts/fankit_synth.py --n 25000
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from fankit_map import mapping  # noqa: E402

DATA = Path("D:/clash-ai-dataset")
RENDERS = DATA / "fankit/characters"
# classes du Fan Kit qu'on ne veut pas (morceaux, personnages hors ladder) ou à renommer vers les classes existantes
DROP = {"goblin-knight", "little-prince-guard", "prince-royal-guard-guard", "zap-barbarian-evolution", "mirror",
        "wizard-void-evolution"}
RENAME = {"tombstone-building-hero": "tombstone-hero", "cannoneer": "cannoneer-tower", "dagger-duchess": "dagger-duchess-tower",
          "elixir-golem": "elixir-golem-big", "freeze": "freeze", "the-log": "the-log"}
SPELLS = {"arrows", "fireball", "zap", "poison", "lightning", "rocket", "freeze", "rage", "clone", "tornado",
          "earthquake", "graveyard", "the-log", "goblin-curse", "vines", "void", "royal-delivery", "giant-snowball"}
BLUE, RED = (255, 150, 30), (40, 40, 235)


def load_render(path: Path) -> np.ndarray | None:
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None or im.ndim != 3 or im.shape[2] != 4:
        return None
    a = im[..., 3]
    if a.mean() > 0.95 * 255:                  # pas détouré (fond plein) : inutilisable
        return None
    ys, xs = np.where(a > 20)
    if len(xs) < 50:
        return None
    return im[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def augment(rgba: np.ndarray, target_h: int, rng: random.Random) -> np.ndarray:
    h, w = rgba.shape[:2]
    squash = rng.uniform(0.78, 0.95)           # vue d'en haut : l'unité paraît plus courte
    scale = target_h / h
    nw, nh = max(8, int(w * scale)), max(8, int(h * scale * squash))
    im = cv2.resize(rgba, (nw, nh), interpolation=cv2.INTER_AREA)
    if rng.random() < 0.5:
        im = im[:, ::-1]
    rgb = im[..., :3].astype(np.float32) * rng.uniform(0.75, 1.15) + rng.uniform(-18, 18)
    im = np.dstack([np.clip(rgb, 0, 255).astype(np.uint8), im[..., 3]])
    if rng.random() < 0.7:
        k = rng.choice([3, 3, 5])
        im = cv2.GaussianBlur(im, (k, k), rng.uniform(0.4, 1.2))
    return im


def paste(img: np.ndarray, rgba: np.ndarray, x0: int, y0: int) -> tuple[int, int, int, int]:
    H, W = img.shape[:2]
    h, w = rgba.shape[:2]
    x1, y1 = min(W, x0 + w), min(H, y0 + h)
    sub = rgba[: y1 - y0, : x1 - x0]
    a = sub[..., 3:4].astype(np.float32) / 255
    img[y0:y1, x0:x1] = (img[y0:y1, x0:x1] * (1 - a) + sub[..., :3] * a).astype(np.uint8)
    return x0, y0, x1, y1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    names_v1 = yaml.safe_load(open(DATA / "data.yaml"))["names"]
    classes = [names_v1[i] for i in sorted(names_v1)]
    # nouvelles classes venues de l'outil d'étiquetage (cartes tapées à la main)
    real_classes = DATA / "real/classes.json"
    extra = json.loads(real_classes.read_text(encoding="utf-8"))[len(classes):] if real_classes.exists() else []
    renders: dict[str, list[np.ndarray]] = {}
    for rid, c in mapping().items():
        c = RENAME.get(c, c)
        if c in DROP:
            continue
        f = next(RENDERS.glob(f"{rid}_*.png"), None)
        im = load_render(f) if f else None
        if im is not None:
            renders.setdefault(c, []).append(im)
    for c in sorted(renders):
        for side in (0, 1):
            if f"{c}_{side}" not in classes:
                extra.append(f"{c}_{side}")
    for k in extra:
        if k not in classes:
            classes.append(k)
    cid = {n: i for i, n in enumerate(classes)}
    # dossier à part : le réglage fin à 300 classes (en cours) ne doit pas voir ces classes >= 300
    (DATA / "images/train_fk").mkdir(parents=True, exist_ok=True)
    (DATA / "labels/train_fk").mkdir(parents=True, exist_ok=True)
    yaml.safe_dump({"path": str(DATA), "train": ["images/train", "images/train_fk"], "val": "images/val",
                    "names": dict(enumerate(classes))},
                   open(DATA / "data_v2.yaml", "w"), sort_keys=False)
    new_units = sorted({k.rsplit("_", 1)[0] for k in classes[len(names_v1):]})
    print(f"{len(renders)} unités avec rendus ; {len(classes)} classes (dont {len(classes) - len(names_v1)} nouvelles) :",
          ", ".join(new_units), flush=True)

    bases = sorted((DATA / "images/train").glob("syn_*.jpg"))
    # les nouvelles unités d'abord (2 chances sur 3), les connues en bonus
    new_pool = [c for c in renders if any(k.startswith(c + "_") for k in classes[len(names_v1):])]
    old_pool = [c for c in renders if c not in new_pool]
    for k in range(a.n):
        base = rng.choice(bases)
        img = cv2.imread(str(base))
        H, W = img.shape[:2]
        labels = [l.split() for l in (DATA / "labels/train" / f"{base.stem}.txt").read_text().splitlines() if l.strip()]
        for _ in range(rng.choice([1, 2, 2, 3])):
            c = rng.choice(new_pool if (rng.random() < 0.67 or not old_pool) else old_pool)
            side = rng.randint(0, 1)
            big = any(w in c for w in ("giant", "golem", "pekka", "machine", "hound", "demolisher", "stein", "bandit"))
            th = rng.randint(70, 150) if big else rng.randint(45, 115)
            if c in SPELLS:
                th = rng.randint(60, 140)
            r = augment(rng.choice(renders[c]), th, rng)
            h, w = r.shape[:2]
            ymin, ymax = (int(0.5 * H), H - h - 60) if side == 0 else (40, int(0.5 * H) - h // 2)
            if rng.random() < 0.2:
                ymin, ymax = 40, H - h - 60                          # parfois de l'autre côté (unité qui attaque)
            x0 = rng.randint(0, max(1, W - w))
            y0 = rng.randint(min(ymin, max(0, ymax)), max(ymin, ymax, 1))
            bx = paste(img, r, x0, y0)
            if c not in SPELLS:                                       # barre de vie du camp, comme en jeu
                bw = max(10, int(0.55 * (bx[2] - bx[0])))
                cx = (bx[0] + bx[2]) // 2
                cv2.rectangle(img, (cx - bw // 2 - 1, bx[1] - 9), (cx + bw // 2 + 1, bx[1] - 3), (20, 20, 20), -1)
                cv2.rectangle(img, (cx - bw // 2, bx[1] - 8), (cx - bw // 2 + int(bw * rng.uniform(0.3, 1)), bx[1] - 4),
                              BLUE if side == 0 else RED, -1)
            # les étiquettes presque entièrement cachées par le collage disparaissent
            keep = []
            for l in labels:
                lx, ly, lw, lh = (float(v) for v in l[1:5])
                l0, l1, l2, l3 = (lx - lw / 2) * W, (ly - lh / 2) * H, (lx + lw / 2) * W, (ly + lh / 2) * H
                ix = max(0, min(l2, bx[2]) - max(l0, bx[0])) * max(0, min(l3, bx[3]) - max(l1, bx[1]))
                if ix < 0.6 * (l2 - l0) * (l3 - l1):
                    keep.append(l)
            labels = keep + [[str(cid[f"{c}_{side}"]), f"{(bx[0] + bx[2]) / 2 / W:.6f}", f"{(bx[1] + bx[3]) / 2 / H:.6f}",
                              f"{(bx[2] - bx[0]) / W:.6f}", f"{(bx[3] - bx[1]) / H:.6f}"]]
        stem = f"fk_{k:06d}"
        cv2.imwrite(str(DATA / "images/train_fk" / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (DATA / "labels/train_fk" / f"{stem}.txt").write_text("\n".join(" ".join(l) for l in labels))
        if (k + 1) % 2500 == 0:
            print(f"{k + 1}/{a.n}", flush=True)


if __name__ == "__main__":
    main()
