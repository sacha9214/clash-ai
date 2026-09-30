"""Ré-étiquetage automatique à DOUBLE VÉRIFICATION des captures de nos matchs (tourne pendant qu'on joue).

Une unité n'est étiquetée que si tout concorde :
  1. deux avis : le scanner voit la même unité au même endroit sur l'image ET sur l'image retournée gauche-droite ;
  2. la couleur du badge (bleu = nous, rouge = eux) confirme le camp, ou les deux détecteurs sont d'accord ;
  3. le jeu : l'unité est dans NOTRE deck (côté bleu) ou dans le deck vu de l'adversaire (côté rouge, sorties
     de générateurs comprises) ; une unité à nous n'apparaît pas pour la première fois chez l'ennemi ;
  4. la partie entière : une unité suivie d'une capture à l'autre garde le même camp.
Une image avec UN SEUL désaccord (unité vue par un seul modèle, camp contradictoire, carte hors deck…) est
écartée entièrement : moins d'images, mais des étiquettes justes (le premier essai avait appris des erreurs).

Sortie : D:/clash-ai-dataset/images/train_fk/dc_<match>_<n>.jpg (+ labels/train_fk), et pour 1 match sur 5
D:/clash-ai-dataset/images/dc_test/ (jeu de test « notre écran », jamais appris).

  .venv-yolo\\Scripts\\python scripts/double_check.py            (tous les matchs pas encore traités)
  .venv-yolo\\Scripts\\python scripts/double_check.py 20260927-145128
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("D:/clash-ai-dataset")
DONE = ROOT / "runs/detector/dc_done.json"
sys.path.insert(0, str(ROOT))
CONF_CONFLICT = 0.12       # une unité vue même faiblement (>= 0.12) sans accord complet = image écartée
CONF_KEEP = 0.35           # une unité étiquetée doit être vue à >= 0.35 par les deux modèles
OWN_UNITS = {"archers": "archer", "musketeer": "musketeer", "minions": "minion", "witch": "witch", "knight": "knight", "valkyrie": "valkyrie", "mini-pekka": "mini-pekka",
             "giant": "giant", "cannon": "cannon"}
WHY: dict = {}                             # raisons de rejet (diagnostic)
LINK = 0.10                                # distance max (fraction de la largeur) entre deux captures (1,5 s)


def is_test_game(game: str) -> bool:
    return int(hashlib.md5(game.encode()).hexdigest(), 16) % 5 == 0


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


from clashai.detect_yolo import DECOR_Y  # noqa: E402


def preds(model, crop, imgsz):
    r = model.predict(crop, imgsz=imgsz, conf=CONF_CONFLICT, verbose=False, device=0)[0]
    out = []
    for box, c, conf in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
        name, side = r.names[int(c)].rsplit("_", 1)
        if (box[1] + box[3]) / 2 < DECOR_Y and "tower" not in name:
            continue                              # décor au-dessus du Roi ennemi (comme en jeu)
        out.append((name, int(side), conf, box))
    return out


def allowed(game: str):
    """Unités possibles de chaque camp : notre deck / le deck adverse vu (+ sorties de ses générateurs)."""
    from clashai.detect_yolo import OWN_NAMES
    from clashai.opponent import SPAWNER_OF, UNIT2CARD
    f = ROOT / "runs/games" / game / "opponent.json"
    if not f.exists():
        return None
    deck = set(json.loads(f.read_text(encoding="utf-8")).get("deck", []))
    enemy = {u for u, (c, *_) in UNIT2CARD.items() if c in deck} | {u for c in deck for u in SPAWNER_OF.get(c, ())}
    return _our_names(game) or OWN_NAMES, enemy


def _our_names(game: str) -> set[str]:
    """Nos unités de CE match, d'après les cartes vues dans notre main (les anciens matchs avaient les Archères)."""
    from clashai.cards import BENCH, DECK
    cards = {**BENCH, **DECK}
    f = ROOT / "runs/games" / game / "decisions.jsonl"
    held = set()
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                held |= {c for c in json.loads(line).get("hand", []) if c in cards}
            except ValueError:
                pass
    units = {u for c in held for u in cards[c].units}
    return units | {u + "-evolution" for u in units}


def process(game: str, det_a, det_b, names_v2: dict) -> tuple[int, int]:
    from clashai import battle as B
    from clashai.detect_yolo import team_color
    ok = allowed(game)
    if ok is None:
        return 0, 0
    own_ok, enemy_ok = ok
    cid = {n: i for i, n in names_v2.items()}
    test = is_test_game(game)
    sub = "dc_test" if test else "train_fk"
    (DATA / "images" / sub).mkdir(parents=True, exist_ok=True)
    (DATA / "labels" / sub).mkdir(parents=True, exist_ok=True)
    dec = ROOT / "runs/games" / game / "decisions.jsonl"
    plays = [json.loads(l) for l in open(dec, encoding="utf-8")] if dec.exists() else []
    plays = [p for p in plays if p.get("ok", True) and p["card"] in OWN_UNITS]
    frames = []                                   # (fichier, crop, unités acceptées, image valable ?)
    for f in sorted(glob.glob(str(ROOT / f"runs/capture/{game}/*.jpg"))):
        img = cv2.imread(f)
        if img is None or not B.in_battle(img):
            continue
        crop, _ = det_a._crop(img)
        pa = preds(det_a.model, crop, det_a.imgsz)
        # 2e avis : le même scanner sur l'image RETOURNÉE gauche-droite (autre point de vue, reste valable quand
        # le scanner progresse ; l'ancien modèle v1 divergeait de plus en plus et faisait tout rejeter)
        W0 = crop.shape[1]
        pb = [(n, sd, c, [W0 - b[2], b[1], W0 - b[0], b[3]])
              for n, sd, c, b in preds(det_a.model, cv2.flip(crop, 1), det_a.imgsz)]
        units, good, used = [], True, set()
        for name, side, conf, box in sorted(pa, key=lambda p: -p[2]):
            j = max((k for k, q in enumerate(pb) if k not in used and q[0] == name and iou(q[3], box) >= 0.4),
                    key=lambda k: iou(pb[k][3], box), default=None)
            if "tower" in name:                   # tours : faciles, on garde l'avis du scanner s'il est sûr
                if conf >= 0.6:
                    units.append((name, side, box))
                if j is not None:
                    used.add(j)
                continue
            if j is None or min(conf, pb[j][2]) < CONF_KEEP:
                good = False; WHY['seul:' + name] = WHY.get('seul:' + name, 0) + 1
                break
            used.add(j)
            color = team_color(crop, tuple(int(v) for v in box))
            sides = {side, pb[j][1]} | ({1 if color > 0 else 0} if color else set())
            if len(sides) != 1:
                good = False; WHY['camp:' + name] = WHY.get('camp:' + name, 0) + 1
                break
            if name not in (enemy_ok if side == 1 else own_ok):
                good = False; WHY['hors deck'] = WHY.get('hors deck', 0) + 1
                break
            units.append((name, side, box))
        if good and any(k not in used and "tower" not in q[0] for k, q in enumerate(pb)):
            good = False; WHY['vu par l ancien seul'] = WHY.get('vu par l ancien seul', 0) + 1
        # nos troupes posées il y a 1 à 8 s doivent être étiquetées : sinon le scanner les a ratées
        # (nos Archères surtout) et l'image apprendrait « rien ici »
        key = int(Path(f).stem)
        for pl in plays:
            dt = (key - int(pl["t"] * 10) % 10**7) / 10
            if 1.0 <= dt <= (25.0 if pl["card"] == "archers" else 8.0) and not any(n == OWN_UNITS[pl["card"]] and sd == 0 for n, sd, _ in units):
                good = False; WHY['notre troupe ratée'] = WHY.get('notre troupe ratée', 0) + 1
        frames.append((f, crop, units, good))
    # 4. cohérence sur la partie : une unité suivie d'une capture à l'autre ne change pas de camp, et une unité
    # à nous ne naît pas chez l'ennemi (au-dessus de la rivière)
    W, H = 568, 896
    prev = []
    for i, (f, crop, units, good) in enumerate(frames):
        cur = []
        for name, side, box in units:
            if "tower" in name:
                continue
            cx, cy = (box[0] + box[2]) / 2 / W, (box[1] + box[3]) / 2 / H
            p = min((q for q in prev if q[0] == name and abs(q[2] - cx) < LINK and abs(q[3] - cy) < LINK * W / H),
                    key=lambda q: abs(q[2] - cx) + abs(q[3] - cy), default=None)
            if p and p[1] != side:
                good = False; WHY['change de camp'] = WHY.get('change de camp', 0) + 1
            if not p and side == 0 and cy < 0.40 and name not in ("archer",):
                good = False; WHY['à nous chez l ennemi'] = WHY.get('à nous chez l ennemi', 0) + 1
            cur.append((name, side, cx, cy))
        prev = cur
        frames[i] = (f, crop, units, good)
    kept = 0
    for f, crop, units, good in frames:
        if not good or not units:
            continue
        lines = []
        for name, side, (x0, y0, x1, y1) in units:
            k = cid.get(f"{name}_{side}")
            if k is None:
                break
            lines.append(f"{k} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}")
        else:
            stem = f"dc_{game}_{Path(f).stem}"
            cv2.imwrite(str(DATA / "images" / sub / f"{stem}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            (DATA / "labels" / sub / f"{stem}.txt").write_text("\n".join(lines))
            kept += 1
    return kept, len(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("games", nargs="*")
    a = ap.parse_args()
    from clashai.detect_yolo import Detector
    names_v2 = yaml.safe_load(open(DATA / "data_v2.yaml"))["names"]
    done = json.loads(DONE.read_text()) if DONE.exists() else {}
    games = a.games or [Path(p).name for p in sorted(glob.glob(str(ROOT / "runs/capture/2026*")))
                        if Path(p).name not in done]
    if not games:
        print("rien de nouveau")
        return
    det_a = Detector(track=False)
    det_b = None
    for g in games:
        kept, total = process(g, det_a, det_b, names_v2)
        done[g] = {"kept": kept, "frames": total, "test": is_test_game(g)}
        DONE.write_text(json.dumps(done, indent=1))
        print(f"{g} : {kept}/{total} images validées ({'test' if is_test_game(g) else 'apprentissage'}) rejets {WHY}", flush=True)
        WHY.clear()


if __name__ == "__main__":
    main()
