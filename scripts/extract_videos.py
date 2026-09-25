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
from clashai.tiles import ARENA as GRID  # noqa: E402

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


def extract(path: str, det: D.Detector, loc: D.Detector) -> dict:
    vid = Path(path).stem
    box = locate_arena(loc, path)
    if box is None:
        return {"video": vid, "error": "arène introuvable"}
    aw, ah = D.ARENA_SIZE
    out = open(OUT / f"{vid}.jsonl", "w", encoding="utf-8")
    battle, last_battle_t, n_frames, n_events = 0, -1e9, 0, 0
    seen_ids, pending, recent, prev = set(), {}, {}, []
    t0 = time.time()
    for t, img in frames(path, FPS):
        n_frames += 1
        arena = crop_arena(img, box)
        units = det.on_arena(arena)
        towers = sum(u.name in ("king-tower", "queen-tower") for u in units)
        if towers < 3:          # menu, écran de fin, webcam plein écran… : pas un combat
            continue
        if t - last_battle_t > 3:   # trou de plus de 3 s (menu, écran de fin) : nouveau combat, on repart de zéro
            battle += 1
            det.tracker.reset()
            seen_ids, pending, recent, prev = set(), {}, {}, []
        last_battle_t = t
        alive = {u.track_id: u for u in units if u.track_id >= 0}
        board = [[u.name, int(u.enemy), round(u.center[0] / aw, 3), round(u.center[1] / ah, 3)]
                 for u in units if u.name not in ("king-tower", "queen-tower")]
        # candidats confirmés : encore là à l'analyse suivante -> une carte posée
        for tid, (card, side, x, y, tc) in list(pending.items()):
            del pending[tid]
            if tid not in alive or t - recent.get((side, card), -1e9) < 1.5:
                continue
            recent[(side, card)] = t
            out.write(json.dumps({"battle": battle, "t": round(tc, 2), "side": side, "card": card,
                                  "x": x, "y": y, "tile": GRID.cell(x, y), "board": board}) + "\n")
            n_events += 1
        for tid, u in alive.items():
            if tid in seen_ids:
                continue
            seen_ids.add(tid)
            if u.name in NOT_UNITS or u.name not in UNIT2CARD:
                continue
            card = UNIT2CARD[u.name][0]
            x, y = u.center[0] / aw, u.center[1] / ah
            # le suivi redonne parfois un numéro neuf à une unité déjà là
            if any(c == card and abs(px - x) < 0.08 and abs(py - y) < 0.06 for c, px, py in prev):
                continue
            side = "opponent" if y < 0.5 else "player"      # on ne pose que dans sa moitié
            pending[tid] = (card, side, round(x, 3), round(y, 3), t)
        prev = [(UNIT2CARD[u.name][0], u.center[0] / aw, u.center[1] / ah) for u in units if u.name in UNIT2CARD]
    out.close()
    return {"video": vid, "arena_box": box, "battles": battle, "frames": n_frames, "events": n_events,
            "video_min": round(duration(path) / 60, 1), "analysis_s": round(time.time() - t0)}


if __name__ == "__main__":
    files = [f for a in sys.argv[1:] for f in glob.glob(a)]
    if not files:
        sys.exit(__doc__)
    OUT.mkdir(parents=True, exist_ok=True)
    det, loc = D.Detector(track=True), D.Detector(track=False)   # suivi pour l'analyse, sans suivi pour localiser
    for f in files:
        r = extract(f, det, loc)
        print(json.dumps(r, ensure_ascii=False), flush=True)
        with open(OUT / "summary.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(r) + "\n")
