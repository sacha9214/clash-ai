"""Cartes que notre détecteur ne connaît pas (sorties après les sprites KataCR d'avril 2024) : préparer leurs exemples.

Liste obtenue en comparant la liste officielle des cartes (deckmelon.com/cards, 123 cartes de base, sept. 2026)
aux 150 unités du détecteur. Héros et évolutions : détectés comme leur carte de base pour l'instant.

Pour chaque nouvelle carte : 2 vidéos de gameplay ciblées -> ~40 images de combat -> pré-étiquetées par YOLO11
(unités connues) -> file de l'outil d'étiquetage, avec l'indice « carte à chercher ». Sacha n'a plus qu'à
encadrer la nouvelle carte et taper son nom (nouvelle classe, voir label_tool.py).

  .venv-yolo\\Scripts\\python scripts/new_cards.py            (téléchargement + images)
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from clashai import detect as D  # noqa: E402
import extract_videos as E  # noqa: E402

NEW_CARDS = ["berserker", "goblinstein", "boss-bandit", "goblin-machine", "ronin", "spirit-empress", "rune-giant",
             "suspicious-bush", "goblin-demolisher", "minion-giant", "goblin-curse", "vines", "void"]
VIDEOS = Path("D:/clash-ai-videos/newcards")
PENDING = Path("D:/clash-ai-dataset/real/pending")
YTDLP = ROOT / ".venv/Scripts/yt-dlp.exe"
DENO = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/WinGet/Packages/DenoLand.Deno_Microsoft.Winget.Source_8wekyb3d8bbwe"
EXCLUDE = ("tier list", "reaction", "#shorts", "news", "sneak peek", "leak", "balance", "2v2", "chaos", "draft")
PER_CARD_VIDEOS, PER_CARD_FRAMES = 2, 40


def env() -> dict:
    return dict(os.environ, PYTHONIOENCODING="utf-8", PATH=str(DENO) + os.pathsep + os.environ.get("PATH", ""))


def find_videos(card: str) -> list[dict]:
    q = f"clash royale {card.replace('-', ' ')} deck gameplay"
    p = subprocess.run([str(YTDLP), f"ytsearch20:{q}", "--flat-playlist", "--print", "%(id)s\t%(duration)s\t%(title)s"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", env=env())
    out = []
    for line in p.stdout.splitlines():
        vid, dur, title = (line.split("\t") + ["", "", ""])[:3]
        words = card.replace("-", " ")
        if dur in ("NA", "None", "") or not (5 * 60 <= float(dur) <= 40 * 60):
            continue
        if words not in title.lower() or any(x in title.lower() for x in EXCLUDE):
            continue                                   # la carte doit être dans le titre : elle sera à l'écran
        out.append({"id": vid, "duration": float(dur), "title": title})
    return out[:PER_CARD_VIDEOS]


def download(card: str, v: dict) -> Path | None:
    d = VIDEOS / card
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{v['id']}.mp4"
    if not f.exists():
        subprocess.run([str(YTDLP), "--no-progress", "--sleep-interval", "5", "-f", "bv*[height<=1080][ext=mp4]/bv*[height<=1080]",
                        "-o", str(d / "%(id)s.%(ext)s"), "--", v["id"]], capture_output=True, env=env())
    return f if f.exists() else None


def queue_frames(card: str, path: Path, det, loc, n: int) -> int:
    box = E.locate_arena(loc, str(path))
    if box is None:
        return 0
    dur = E.duration(str(path))
    times = sorted(random.sample(range(int(dur * 0.1), int(dur * 0.95), 3), min(n * 3, int(dur * 0.85) // 3)))
    aw, ah = D.ARENA_SIZE
    done = 0
    for t in times:
        if done >= n:
            break
        for _, img in E.frames(str(path), 1, t, t + 1.5):
            crop = E.crop_arena(img, box)
            units = det.on_arena(crop)
            if sum(u.name in ("king-tower", "queen-tower") for u in units) < 3:
                break                                  # pas un combat (menu, webcam…)
            stem = f"new_{card}_{path.stem}_{t:05d}"
            boxes = [{"name": u.name, "side": int(u.enemy), "conf": round(u.conf, 2),
                      "box": [round(u.box[0] / aw, 4), round(u.box[1] / ah, 4), round(u.box[2] / aw, 4), round(u.box[3] / ah, 4)]}
                     for u in units]
            cv2.imwrite(str(PENDING / f"{stem}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            (PENDING / f"{stem}.json").write_text(json.dumps(boxes))
            (PENDING / f"{stem}.hint").write_text(card)
            done += 1
            break
    return done


def main():
    random.seed(0)
    PENDING.mkdir(parents=True, exist_ok=True)
    det, loc = D.Detector(track=False), D.Detector(track=False)
    report = {}
    for card in NEW_CARDS:
        vids = find_videos(card)
        files = [f for f in (download(card, v) for v in vids) if f]
        n = sum(queue_frames(card, f, det, loc, PER_CARD_FRAMES // max(1, len(files))) for f in files)
        report[card] = {"videos": len(files), "frames": n}
        print(f"{card:18s} vidéos {len(files)}  images {n}", flush=True)
    (VIDEOS / "report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
