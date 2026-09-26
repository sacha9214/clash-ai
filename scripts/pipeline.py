"""Pipeline complet, autonome (processus Windows, tourne même si Claude est fermé) :

1. téléchargement des vidéos jusqu'à --hours (en parallèle de l'analyse) ;
2. analyse des vidéos avec NOTRE détecteur YOLO11 TensorRT, --workers vidéos en même temps ;
3. comparaison pros / IA + réentraînement du modèle de placement ;
4. réglage fin du détecteur (points faibles) ;
5. entraînement v2 (nouvelles cartes, héros, évolutions) ;
6. pseudo-étiquetage des images des nouvelles cartes avec v2 ;
7. rapports poussés sur GitHub après chaque étape (code, rapports, petits modèles).

  .venv\\Scripts\\python scripts/pipeline.py --hours 100 --workers 4
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
LOG = ROOT / "runs/pipeline.log"
PY = ROOT / ".venv/Scripts/python.exe"
PY_YOLO = ROOT / ".venv-yolo/Scripts/python.exe"
VIDEOS = Path("D:/clash-ai-videos/1080p")
MANIFEST = Path("D:/clash-ai-videos/manifest.jsonl")


def log(msg: str) -> None:
    line = f"[{time.strftime('%d/%m %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def env(**extra) -> dict:
    return dict(os.environ, PYTHONIOENCODING="utf-8", **extra)


def step(name: str, cmd: list, out: Path, **kw) -> int:
    log(f"--- {name} ---")
    t = time.time()
    with open(out, "w", encoding="utf-8") as f:
        code = subprocess.run([str(c) for c in cmd], cwd=ROOT, env=env(**kw), stdout=f, stderr=subprocess.STDOUT).returncode
    log(f"{name} : terminé en {(time.time() - t) / 60:.0f} min (code {code})")
    return code


def to_analyze() -> list[str]:
    ids = [json.loads(l)["id"] for l in open(MANIFEST, encoding="utf-8")] if MANIFEST.exists() else []
    return [v for v in dict.fromkeys(ids)
            if (VIDEOS / f"{v}.mp4").exists() and not (ROOT / f"runs/videos/{v}.battles.json").exists()]   # une analyse
    # interrompue (fichier brut sans résultat) est simplement refaite


def github(msg: str) -> None:
    """Pousse le code, les rapports et les petits modèles (pas les vidéos ni les jeux de données)."""
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    for src in ["runs/detector/NIGHT_REPORT.md", "runs/detector/FINETUNE_REPORT.md", "runs/detector/V2_REPORT.md",
                "runs/detector/pseudo_labels.json", "runs/videos/placement_report.txt", "runs/videos/placement_score.jsonl",
                "runs/detector/eval.jsonl"]:
        s = ROOT / src
        if s.exists():
            (results / s.name).write_bytes(s.read_bytes())
    for cmd in (["git", "add", "-A", "clashai", "scripts", "learning", "results", "HANDOFF.md", "README.md"],
                ["git", "add", "-f", "models/placement/model.pt", "models/placement/meta.json", "models/yolo/clashai_yolo11s.pt"],
                ["git", "commit", "-q", "-m", f"{msg}\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"],
                ["git", "push", "-q", "origin", "main"]):
        subprocess.run(cmd, cwd=ROOT, capture_output=True)
    log(f"GitHub : {msg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=100)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-download", action="store_true")
    a = ap.parse_args()
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    log(f"=== pipeline : {a.hours} h de vidéos, {a.workers} analyses en parallèle ===")

    # 1-2. téléchargement et analyse en parallèle
    dl = None if a.skip_download else subprocess.Popen(
        [str(PY), "scripts/night_videos.py", "--hours", str(a.hours), "--download-only"], cwd=ROOT, env=env(),
        stdout=open(ROOT / "runs/videos/download.out.txt", "w"), stderr=subprocess.STDOUT)
    running: dict[str, subprocess.Popen] = {}
    tried: set[str] = set()          # une vidéo sans arène (conseils, menus…) n'est tentée qu'une fois
    done = 0
    t0 = time.time()
    while True:
        for vid, p in list(running.items()):
            if p.poll() is not None:
                del running[vid]
                done += 1
        todo = [v for v in to_analyze() if v not in running and v not in tried]
        while todo and len(running) < a.workers:
            vid = todo.pop(0)
            tried.add(vid)
            running[vid] = subprocess.Popen([str(PY_YOLO), "scripts/extract_videos.py", str(VIDEOS / f"{vid}.mp4")],
                                            cwd=ROOT, env=env(), stdout=open(ROOT / f"runs/videos/{vid}.log", "w"),
                                            stderr=subprocess.STDOUT)
        if not running and not todo and (dl is None or dl.poll() is not None):
            break
        if done and done % 10 == 0:
            log(f"analyse : {done} vidéos en {(time.time() - t0) / 60:.0f} min, {len(todo)} en attente")
        time.sleep(20)
    log(f"analyse terminée : {done} vidéos en {(time.time() - t0) / 60:.0f} min")

    # 3. placement : comparaison + réentraînement (CPU)
    step("comparaison pros / IA", [PY_YOLO, "scripts/compare_placements.py"], ROOT / "runs/videos/placement_report.txt")
    step("modèle de placement", [PY_YOLO, "scripts/train_placement.py"], ROOT / "runs/videos/placement_train.txt",
         CUDA_VISIBLE_DEVICES="")
    github("Placement model retrained on all analyzed videos")

    # 4-6. détecteur
    step("réglage fin du détecteur", [PY_YOLO, "scripts/finetune_detector.py", "--hours", "4"], ROOT / "runs/detector/finetune.out.txt")
    github("Detector fine-tuned on its weak spots")
    step("entraînement v2", [PY_YOLO, "scripts/train_v2.py", "--hours", "8"], ROOT / "runs/detector/train_v2.out.txt")
    github("Detector v2: new cards, heroes and evolutions")
    found = sorted(ROOT.glob("runs/**/yolo11s_cr_v2/weights/best.pt"), key=lambda p: p.stat().st_mtime)
    if found:
        step("pseudo-étiquetage", [PY_YOLO, "scripts/pseudo_label.py", "--model", found[-1]], ROOT / "runs/detector/pseudo_label.txt")
        github("Pseudo-labels for new cards (to be checked)")
    log("=== pipeline : fin ===")


if __name__ == "__main__":
    main()
