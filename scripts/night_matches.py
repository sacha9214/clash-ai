"""Matchs automatiques de nuit (Windows), une fois le GPU libéré par l'entraînement du détecteur.

Lots de 5 combats (autoplay.py, avec apprentissage des stratégies) jusqu'à --games combats.
Arrêt propre : 3 échecs de suite (écran inconnu, plantage, téléphone débranché), ou fichier runs/games/STOP.
Pas de réclamation de récompenses ni d'amélioration de cartes (pas encore testées sur ce PC).

  .venv-katacr\\Scripts\\python scripts/night_matches.py --games 30 --wait-pid 12345
"""
from __future__ import annotations

import argparse
import ctypes
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "runs/games/night_matches.log"
JOURNAL = ROOT / "runs/games/journal.jsonl"
STOP = ROOT / "runs/games/STOP"
PY = ROOT / ".venv-yolo/Scripts/python.exe"   # notre détecteur YOLO11 TensorRT


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def pid_alive(pid: int) -> bool:
    return str(pid) in subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True).stdout


def games_played() -> int:
    return sum(1 for _ in open(JOURNAL, encoding="utf-8")) if JOURNAL.exists() else 0


def phone_ok() -> bool:
    sys.path.insert(0, str(ROOT))
    from clashai.device import ADB
    out = subprocess.run([ADB, "devices"], capture_output=True, text=True).stdout
    if "offline" in out:
        subprocess.run([ADB, "reconnect", "offline"], capture_output=True)
        time.sleep(5)
        out = subprocess.run([ADB, "devices"], capture_output=True, text=True).stdout
    return any(line.endswith("\tdevice") for line in out.splitlines())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--wait-pid", type=int, nargs="*", default=[], help="attendre la fin de ces processus (GPU occupé)")
    a = ap.parse_args()
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    for pid in a.wait_pid:
        log(f"attente de la fin du processus {pid} (GPU occupé)…")
        while pid_alive(pid):
            time.sleep(60)
    start, fails = games_played(), 0
    log(f"=== matchs de nuit : objectif {a.games} combats ===")
    while games_played() - start < a.games and not STOP.exists() and fails < 3:
        if not phone_ok():
            fails += 1
            log(f"téléphone non détecté par adb ({fails}/3), nouvel essai dans 2 min")
            time.sleep(120)
            continue
        n = min(5, a.games - (games_played() - start))
        before = games_played()
        p = subprocess.run([str(PY), "scripts/autoplay.py", "--games", str(n)], cwd=ROOT, capture_output=True,
                           text=True, encoding="utf-8", errors="replace", env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(p.stdout[-5000:] + "\n" + p.stderr[-3000:] + "\n")
        done = games_played() - before
        results = [l for l in p.stdout.splitlines() if "'result'" in l]
        wins = sum("'result': 'win'" in l for l in results)
        log(f"lot terminé : {done} combats ({wins} victoires), total {games_played() - start}/{a.games}")
        if done == 0 or "écran inconnu" in p.stdout or p.returncode != 0:
            fails += 1
            log(f"problème ({fails}/3) : code {p.returncode}, pause 2 min")
            time.sleep(120)
        else:
            fails = 0
    reason = "objectif atteint" if games_played() - start >= a.games else "STOP" if STOP.exists() else "3 échecs de suite"
    log(f"=== matchs de nuit : fin ({reason}), {games_played() - start} combats ===")


if __name__ == "__main__":
    main()
