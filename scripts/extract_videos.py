"""Regarde des vidéos YouTube de gameplay et en tire chaque carte jouée (phase 3 : imitation).

1. Localiser l'arène : la vidéo contient souvent le jeu dans une colonne (webcam, logos autour).
   On essaie des fenêtres pleine hauteur, on garde celle où l'on voit les deux tours du Roi,
   puis on cale le recadrage 568x896 pour que les Rois tombent là où ils sont dans les images
   d'entraînement de KataCR.
2. Analyser ~5 images/s (bien plus rapide que le temps réel) : suivi des unités, et chaque
   unité qui APPARAÎT dans une moitié = une carte posée par ce camp (on ne pose que chez soi).
3. Écrire runs/videos/<id>.jsonl : combat, instant, camp, carte, position (fractions de
   l'arène), et le plateau à ce moment-là.

  .venv-katacr\\Scripts\\python scripts/extract_videos.py D:/clash-ai-videos/1080p/*.mp4
"""
from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

import av
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from clashai import detect as D  # noqa: E402
from clashai.cards import NOT_UNITS  # noqa: E402
from clashai.opponent import UNIT2CARD  # noqa: E402
from clashai.tiles import ARENA as GRID, Grid  # noqa: E402

FPS = 5.0
# Rois dans l'arène 568x896 d'entraînement (médiane des étiquettes KataCR)
KING_Y_ENEMY, KING_Y_OWN = 0.127, 0.874
OUT = ROOT / "runs/videos"


def frames(path: str, fps: float, start: float = 0.0, end: float | None = None):
    """(instant, image BGR) à ~fps images/s, sans convertir les images sautées."""
    c = av.open(path)
    s = c.streams.video[0]
    s.thread_type = "AUTO"
    if start:
        c.seek(int(start / s.time_base), stream=s)
    nxt = start
    for f in c.decode(s):
        t = float(f.pts * s.time_base) if f.pts is not None else nxt
        if end is not None and t > end:
            break
        if t + 1e-3 >= nxt:
            nxt = t + 1 / fps
            yield t, f.to_ndarray(format="bgr24")
    c.close()


def duration(path: str) -> float:
    with av.open(path) as c:
        return float(c.duration / 1e6)


def locate_arena(det: D.Detector, path: str) -> tuple[int, int, int, int] | None:
    """Boîte (x0, y0, w, h) de l'arène dans la vidéo, ou None si aucun combat trouvé."""
    dur = duration(path)
    samples = []
    for k in range(24):
        t = dur * (k + 0.5) / 24
        for _, img in frames(path, 1, t, t + 1.5):
            samples.append(img)
            break
    H, W = samples[0].shape[:2]
    best = (0, None)
    aw, ah = D.ARENA_SIZE
    for aspect in (0.45, 0.5, 0.5625, 0.62):
        w = int(H * aspect)
        for x0 in range(0, max(1, W - w + 1), max(8, W // 60)):
            score, kings = 0, {True: [], False: []}   # True = Roi du haut (ennemi), selon la position seule
            sx, sy = w / aw, H / ah
            for img in samples[::3]:
                # fenêtre entière (pas le recadrage du téléphone : chaque vidéo a sa mise en page)
                for u in det.on_arena(cv2.resize(img[:, x0:x0 + w], D.ARENA_SIZE)):
                    if u.name in ("king-tower", "queen-tower") and u.conf > 0.6:
                        score += 1
                        if u.name == "king-tower":
                            # le camp deviné par le détecteur n'est pas fiable sur les vidéos : on prend la position
                            kings[u.center[1] < ah / 2].append((x0 + u.center[0] * sx, u.center[1] * sy))
            if score > best[0] and kings[True] and kings[False]:
                best = (score, (kings, x0, w))
    if best[1] is None:
        return None
    kings, x0, w = best[1]
    (ex, ey), (_, oy) = np.median(kings[True], 0), np.median(kings[False], 0)
    ah = (oy - ey) / (KING_Y_OWN - KING_Y_ENEMY)
    aw = ah * D.ARENA_SIZE[0] / D.ARENA_SIZE[1]
    return int(ex - aw / 2), int(ey - KING_Y_ENEMY * ah), int(aw), int(ah)


def crop_arena(img: np.ndarray, box) -> np.ndarray:
    x0, y0, w, h = box
    H, W = img.shape[:2]
    pad = cv2.copyMakeBorder(img, max(0, -y0), max(0, y0 + h - H), max(0, -x0), max(0, x0 + w - W),
                             cv2.BORDER_CONSTANT)
    x0, y0 = max(0, x0), max(0, y0)
    return cv2.resize(pad[y0:y0 + h, x0:x0 + w], D.ARENA_SIZE, interpolation=cv2.INTER_AREA)


TOWERS = ("king-tower", "queen-tower")


def detect_raw(path: str, det: D.Detector, loc: D.Detector) -> dict:
    """Passe GPU : détecte tout, image par image, et l'enregistre tel quel (runs/videos/<id>.raw.jsonl).
    Le découpage en combats, les cartes et le vainqueur se calculent ensuite sans GPU (process)."""
    vid = Path(path).stem
    box = locate_arena(loc, path)
    if box is None:
        return {"video": vid, "error": "arène introuvable"}
    aw, ah = D.ARENA_SIZE
    t0, last_ok, n = time.time(), -1e9, 0
    with open(OUT / f"{vid}.raw.jsonl", "w", encoding="utf-8") as out:
        out.write(json.dumps({"video": vid, "arena_box": box, "fps": FPS}) + "\n")
        for t, img in frames(path, FPS):
            n += 1
            units = det.on_arena(crop_arena(img, box))
            ok = sum(u.name in TOWERS for u in units) >= 3
            if ok and t - last_ok > 3:
                det.tracker.reset()          # coupure : le suivi repart propre
            last_ok = t if ok else last_ok
            out.write(json.dumps({"t": round(t, 2), "u": [
                [u.track_id, u.name, int(u.enemy), round(u.center[0] / aw, 4), round(u.center[1] / ah, 4), round(u.conf, 2)]
                for u in units]}) + "\n")
    return {"video": vid, "arena_box": box, "frames": n, "video_min": round(duration(path) / 60, 1),
            "analysis_s": round(time.time() - t0)}


def fit_grid(frames_: list) -> Grid:
    """Grille propre à la vidéo, calée sur ses tours (princesses rangées 6.5 / 25.5, colonnes 3.5 / 14.5 ;
    Rois rangées 2.5 / 29.5, colonne 9). Corrige le décalage des vieilles vidéos."""
    px, py = [], []
    for f in frames_:
        for _, name, _, x, y, conf in f["u"]:
            if name not in TOWERS or conf < 0.6:
                continue
            top = y < 0.5
            if name == "queen-tower":
                px.append((x, 3.5 if x < 0.5 else 14.5))
                py.append((y, 6.5 if top else 25.5))
            else:
                px.append((x, 9.0))
                py.append((y, 2.5 if top else 29.5))
    if len({c for _, c in py}) < 2 or len({c for _, c in px}) < 2:
        return GRID
    fx = np.polyfit([c for _, c in px], [v for v, _ in px], 1)   # x = x0 + tw * colonne
    fy = np.polyfit([c for _, c in py], [v for v, _ in py], 1)
    return Grid(x0=float(fx[1]), tw=float(fx[0]), y0=float(fy[1]), th=float(fy[0]))


def process(vid: str) -> dict:
    """Passe sans GPU : combats, cartes jouées (filtrées par deck), vainqueur, grille de la vidéo."""
    lines = open(OUT / f"{vid}.raw.jsonl", encoding="utf-8").read().splitlines()
    F = [json.loads(line) for line in lines[1:]]
    for f in F:
        f["top"] = sum(1 for u in f["u"] if u[1] in TOWERS and u[4] < 0.5)
        f["bot"] = sum(1 for u in f["u"] if u[1] in TOWERS and u[4] >= 0.5)
        f["busy"] = sum(1 for u in f["u"] if u[1] not in TOWERS and u[1] not in NOT_UNITS)
        f["ok"] = f["top"] + f["bot"] >= 3
    grid = fit_grid([f for f in F if f["ok"]])

    # --- découpage en combats ---
    battle, last_ok, hist = 0, None, []
    for i, f in enumerate(F):
        if not f["ok"]:
            f["b"] = 0
            continue
        new = battle == 0
        if last_ok is not None and f["t"] - F[last_ok]["t"] > 3:
            # coupure : nouveau combat seulement si le plateau repart vide (sinon plan de coupe en plein match)
            nxt = [g["busy"] for g in F[i:i + 10] if g["ok"]]
            new = new or (bool(nxt) and float(np.mean(nxt)) <= 1.0)
        stable = int(np.median(hist[-10:])) if len(hist) >= 10 else 6
        ahead = [g["top"] + g["bot"] for g in F[i:i + 10] if g["ok"]]
        if stable <= 5 and len(ahead) >= 8 and min(ahead) >= 6:
            new = True                      # les 6 tours réapparaissent : on est dans le combat suivant
        if new:
            battle += 1
            hist = []
        f["b"] = battle
        hist.append(f["top"] + f["bot"])
        last_ok = i

    # --- cartes posées : une unité qui APPARAÎT dans une moitié = une carte de ce camp ---
    events, seen_ids, pending, recent, prev, cur = [], set(), {}, {}, [], 0
    for f in F:
        if not f["b"]:
            continue
        if f["b"] != cur:
            cur, seen_ids, pending, recent, prev = f["b"], set(), {}, {}, []
        alive = {u[0]: u for u in f["u"] if u[0] >= 0}
        board = [[u[1], u[2], u[3], u[4]] for u in f["u"] if u[1] not in TOWERS]
        for tid, (card, side, x, y, tc) in list(pending.items()):
            del pending[tid]
            if tid in alive and f["t"] - recent.get((side, card), -1e9) >= 1.5:
                recent[(side, card)] = f["t"]
                events.append({"battle": cur, "t": tc, "side": side, "card": card, "x": x, "y": y,
                               "tile": grid.cell(x, y), "board": board})
        for tid, (_, name, _, x, y, _) in alive.items():
            if tid in seen_ids:
                continue
            seen_ids.add(tid)
            if name in NOT_UNITS or name not in UNIT2CARD:
                continue
            card = UNIT2CARD[name][0]
            if any(c == card and abs(qx - x) < 0.08 and abs(qy - y) < 0.06 for c, qx, qy in prev):
                continue                    # le suivi redonne parfois un numéro neuf à une unité déjà là
            pending[tid] = (card, "opponent" if y < 0.5 else "player", x, y, f["t"])
        prev = [(UNIT2CARD[u[1]][0], u[3], u[4]) for u in f["u"] if u[1] in UNIT2CARD]

    # --- par combat : deck (8 cartes les plus vues, au moins 2 fois), vainqueur (tours restantes à la fin) ---
    battles = []
    for b in range(1, battle + 1):
        fb = [f for f in F if f["b"] == b]
        if not fb or fb[-1]["t"] - fb[0]["t"] < 30:
            continue                        # trop court pour être un vrai combat
        info = {"battle": b, "start": fb[0]["t"], "end": fb[-1]["t"], "decks": {}}
        for side in ("player", "opponent"):
            cnt = {}
            for e in events:
                if e["battle"] == b and e["side"] == side:
                    cnt[e["card"]] = cnt.get(e["card"], 0) + 1
            info["decks"][side] = [c for c, k in sorted(cnt.items(), key=lambda kv: -kv[1]) if k >= 2][:8]
        tail = [f for f in fb if f["t"] >= fb[-1]["t"] - 5]
        top, bot = float(np.median([f["top"] for f in tail])), float(np.median([f["bot"] for f in tail]))
        info["winner"] = "player" if bot > top else "opponent" if top > bot else None
        info["towers_end"] = {"player": bot, "opponent": top}
        battles.append(info)
    keep = {(b["battle"], s): set(b["decks"][s]) for b in battles for s in ("player", "opponent")}
    winner = {b["battle"]: b["winner"] for b in battles}
    clean = [e for e in events if e["card"] in keep.get((e["battle"], e["side"]), set())]
    with open(OUT / f"{vid}.jsonl", "w", encoding="utf-8") as fh:
        for e in clean:
            e["winner"] = winner[e["battle"]]
            fh.write(json.dumps(e) + "\n")
    with open(OUT / f"{vid}.battles.json", "w", encoding="utf-8") as fh:
        json.dump({"video": vid, "grid": grid.__dict__, "battles": battles}, fh, indent=1)
    return {"video": vid, "battles": len(battles), "events_raw": len(events), "events": len(clean),
            "winners": [b["winner"] for b in battles]}


if __name__ == "__main__":
    # --process : recalcule à partir des .raw.jsonl déjà faits, sans GPU
    only_process = "--process" in sys.argv
    files = [f for a in sys.argv[1:] if a != "--process" for f in glob.glob(a)]
    OUT.mkdir(parents=True, exist_ok=True)
    if only_process:
        for f in files or glob.glob(str(OUT / "*.raw.jsonl")):
            print(json.dumps(process(Path(f).name.split(".")[0]), ensure_ascii=False), flush=True)
        sys.exit()
    if not files:
        sys.exit(__doc__)
    det, loc = D.Detector(track=True), D.Detector(track=False)   # suivi pour l'analyse, sans suivi pour localiser
    for f in files:
        r = detect_raw(f, det, loc)
        if "error" not in r:
            r.update(process(r["video"]))
        print(json.dumps(r, ensure_ascii=False), flush=True)
        with open(OUT / "summary.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(r) + "\n")
