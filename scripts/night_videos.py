"""Nuit des vidéos, autonome : cherche et télécharge des parties récentes et variées, puis (une fois le GPU
libéré par l'entraînement du détecteur) les analyse et compare nos placements à ceux des joueurs.

Accord de Sacha (2026-09-25) : télécharger beaucoup plus de vidéos, de tous les points de vue.
Critères : parties récentes (2025+), 8 à 90 min, 1080p max, decks et niveaux variés ; pas de tier lists,
réactions, shorts. Plafond : --hours d'heures de vidéo.

  .venv\\Scripts\\python scripts/night_videos.py --hours 30 --wait-pid 12345
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path("D:/clash-ai-videos/1080p")
LOG = ROOT / "runs/videos/night_videos.log"
MANIFEST = Path("D:/clash-ai-videos/manifest.jsonl")
YTDLP = ROOT / ".venv/Scripts/yt-dlp.exe"
PY_KATACR = ROOT / ".venv-katacr/Scripts/python.exe"
DENO_DIR = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/WinGet/Packages/DenoLand.Deno_Microsoft.Winget.Source_8wekyb3d8bbwe"

# Points de vue variés : decks, niveaux (dont bas niveau, comme notre compte), top ladder
QUERIES = [
    "clash royale gameplay no commentary 2026", "clash royale top ladder gameplay 2026",
    "clash royale giant deck gameplay", "clash royale giant beatdown gameplay", "clash royale hog rider cycle gameplay",
    "clash royale golem beatdown gameplay", "clash royale log bait gameplay", "clash royale lava hound gameplay",
    "clash royale x-bow gameplay", "clash royale royal giant gameplay", "clash royale pekka bridge spam gameplay",
    "clash royale mega knight gameplay", "clash royale graveyard gameplay", "clash royale miner control gameplay",
    "clash royale arena 2 gameplay", "clash royale low arena gameplay", "clash royale arena 4 gameplay",
    "clash royale ladder push gameplay 2025", "clash royale path of legends gameplay", "clash royale ranked gameplay 2026",
    "clash royale 1v1 ladder full match", "clash royale pro player gameplay",
]
EXCLUDE = ("tier list", "reaction", "#shorts", "tutorial", "guide", "top 10", "best decks", "funny", "moments",
           "tips", "compilation", "news", "update", "sneak peek", "balance",
           "chaos", "2v2", "draft", "touchdown", "triple elixir", "event", "challenge")   # modes spéciaux : autre jeu


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def env() -> dict:
    return dict(os.environ, PYTHONIOENCODING="utf-8", PATH=str(DENO_DIR) + os.pathsep + os.environ.get("PATH", ""))


def search(query: str, n: int = 25) -> list[dict]:
    """Recherche YouTube (métadonnées seulement)."""
    p = subprocess.run([str(YTDLP), f"ytsearch{n}:{query}", "--flat-playlist", "--print",
                        "%(id)s\t%(duration)s\t%(channel)s\t%(title)s"], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env())
    out = []
    for line in p.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) == 4 and parts[1] not in ("NA", "None", ""):
            out.append({"id": parts[0], "duration": float(parts[1]), "channel": parts[2], "title": parts[3], "query": query})
    return out


def pick(hours: float) -> list[dict]:
    """Tour à tour entre les recherches (une vidéo par deck / niveau à chaque tour) : tous les points de vue
    sont représentés, au lieu que les premières recherches prennent tout le quota."""
    have = {p.stem for p in OUT.glob("*.mp4")}
    pools = []
    for q in QUERIES:
        ok = [v for v in search(q) if 8 * 60 <= v["duration"] <= 90 * 60
              and not any(x in v["title"].lower() for x in EXCLUDE)]
        pools.append(ok)
        log(f"  recherche « {q} » : {len(ok)} candidates")
    chosen, seen, per_channel, total = [], set(have), {}, 0.0
    while total < hours and any(pools):
        for pool in pools:
            while pool:
                v = pool.pop(0)
                if v["id"] in seen or per_channel.get(v["channel"], 0) >= 4:   # max 4 par chaîne : joueurs variés
                    continue
                seen.add(v["id"])
                per_channel[v["channel"]] = per_channel.get(v["channel"], 0) + 1
                chosen.append(v)
                total += v["duration"] / 3600
                break
            if total >= hours:
                break
    return chosen


def download(v: dict) -> bool:
    p = subprocess.run([str(YTDLP), "--no-progress", "--sleep-interval", "5", "--write-info-json",
                        "-f", "bv*[height<=1080][ext=mp4]/bv*[height<=1080]", "-o", str(OUT / "%(id)s.%(ext)s"),
                        "--", v["id"]], capture_output=True, text=True, encoding="utf-8", errors="replace", env=env())
    ok = (OUT / f"{v['id']}.mp4").exists()
    if not ok:
        log(f"  échec {v['id']} : {p.stderr.strip().splitlines()[-1:] }")
    return ok


def pid_alive(pid: int) -> bool:
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True).stdout
    return str(pid) in out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=30.0)
    ap.add_argument("--wait-pid", type=int, default=None, help="attendre la fin de ce processus (entraînement) avant le GPU")
    a = ap.parse_args()
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"=== nuit des vidéos : objectif {a.hours} h ===")
    chosen = pick(a.hours)
    log(f"{len(chosen)} vidéos retenues, {sum(v['duration'] for v in chosen) / 3600:.1f} h")
    new = []
    for i, v in enumerate(chosen, 1):
        log(f"[{i}/{len(chosen)}] {v['duration'] / 60:.0f} min | {v['channel']} | {v['title'][:70]}")
        if download(v):
            new.append(v["id"])
            with open(MANIFEST, "a", encoding="utf-8") as f:
                f.write(json.dumps(v, ensure_ascii=False) + "\n")
    log(f"téléchargées : {len(new)}")

    if a.wait_pid:
        log(f"attente de la fin de l'entraînement (PID {a.wait_pid}) pour utiliser le GPU…")
        while pid_alive(a.wait_pid):
            time.sleep(60)
    if new:
        log("analyse des nouvelles vidéos (GPU)…")
        files = [str(OUT / f"{vid}.mp4") for vid in new]
        subprocess.run([str(PY_KATACR), "scripts/extract_videos.py", *files], cwd=ROOT, env=env(),
                       stdout=open(LOG, "a", encoding="utf-8"), stderr=subprocess.STDOUT)
    log("comparaison de nos placements avec ceux des joueurs…")
    subprocess.run([str(PY_KATACR), "scripts/compare_placements.py"], cwd=ROOT, env=env(),
                   stdout=open(ROOT / "runs/videos/placement_report.txt", "w", encoding="utf-8"), stderr=subprocess.STDOUT)
    log("=== nuit des vidéos : fin (rapport : runs/videos/placement_report.txt) ===")


if __name__ == "__main__":
    main()
