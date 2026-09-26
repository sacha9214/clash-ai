"""Réglage fin du détecteur sur ses points faibles, autonome (tourne même si Claude est fermé).

Repart du modèle actuel (models/yolo/clashai_yolo11s.pt) avec en plus 15 000 images centrées sur les unités
qu'il rate (gen_synthetic.py --only ...) et, s'il y en a, les images réelles corrigées dans label_tool.py.
Remplace le détecteur de l'IA SEULEMENT s'il fait mieux sur le jeu de test (séquences jamais vues).

  .venv-yolo\Scripts\python scripts/finetune_detector.py --hours 4 --wait-pid 123 456
"""
from __future__ import annotations

import argparse
import ctypes
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from night_detector import PY_YOLO, last_eval, log, run  # noqa: E402
import night_detector  # noqa: E402

MODELS = ROOT / "models/yolo"
NAME = "yolo11s_cr_ft"
night_detector.LOG = ROOT / "runs/detector/finetune.log"


def pid_alive(pid: int) -> bool:
    return str(pid) in subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True).stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=4.0)
    ap.add_argument("--wait-pid", type=int, nargs="*", default=[])
    a = ap.parse_args()
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    for pid in a.wait_pid:
        log(f"attente de la fin du processus {pid} (GPU ou images en cours)…")
        while pid_alive(pid):
            time.sleep(60)
    log("=== réglage fin : début ===")
    # images réelles corrigées dans label_tool.py : ajoutées à l'entraînement (les boîtes de NOUVELLES cartes,
    # classes >= 300, attendront un modèle élargi : on les retire ici)
    real, n = Path("D:/clash-ai-dataset/real"), 0
    for lab in sorted((real / "labels").glob("*.txt")) if (real / "labels").exists() else []:
        img = real / "images" / f"{lab.stem}.jpg"
        if img.exists():
            keep = [l for l in lab.read_text().splitlines() if l.strip() and int(l.split()[0]) < 300]
            (Path("D:/clash-ai-dataset/labels/train") / f"fix_{lab.stem}.txt").write_text("\n".join(keep))
            shutil.copy(img, Path("D:/clash-ai-dataset/images/train") / f"fix_{lab.stem}.jpg")
            n += 1
    log(f"images réelles corrigées ajoutées : {n}")
    code, out = run([PY_YOLO, "scripts/eval_yolo.py", MODELS / "clashai_yolo11s.engine"])
    before = last_eval(out)
    log(f"avant : F1 {before and before['f1']}")
    run([PY_YOLO, "scripts/train_yolo.py", "--model", MODELS / "clashai_yolo11s.pt", "--hours", a.hours, "--lr0", "0.002",
         "--close-mosaic", "2", "--workers", "11", "--name", NAME])
    found = sorted(ROOT.glob(f"runs/**/{NAME}/weights/best.pt"), key=lambda p: p.stat().st_mtime)
    if not found:
        log("pas de modèle : voir le journal")
        return
    best = found[-1]
    run([PY_YOLO, "-c", f"from ultralytics import YOLO; YOLO(r'{best}').export(format='engine', half=True, imgsz=896, device=0)"])
    engine = best.with_suffix(".engine")
    code, out = run([PY_YOLO, "scripts/eval_yolo.py", engine if engine.exists() else best])
    after = last_eval(out)
    better = bool(before and after and after["f1"] > before["f1"])
    if better and engine.exists():
        for ext in (".pt", ".engine"):                      # l'ancien est gardé sous _v1
            shutil.copy(MODELS / f"clashai_yolo11s{ext}", MODELS / f"clashai_yolo11s_v1{ext}")
        shutil.copy(best, MODELS / "clashai_yolo11s.pt")
        shutil.copy(engine, MODELS / "clashai_yolo11s.engine")
    rows = [("Avant (YOLO11 TensorRT)", before), ("Après réglage fin", after)]
    lines = ["# Réglage fin du détecteur", "", "| Modèle | Précision | Rappel | F1 | Camp | ms/image |", "|---|---|---|---|---|---|"]
    for n, r in rows:
        lines.append(f"| {n} | {r['precision']:.1%} | {r['recall']:.1%} | **{r['f1']:.3f}** | {r['side_accuracy']:.1%} | {r['ms_per_image']} |"
                     if r else f"| {n} | — | — | — | — | — |")
    lines += ["", f"**{'Meilleur : l’IA utilise maintenant le modèle réglé' if better else 'Pas meilleur : l’IA garde le modèle actuel'}.**"]
    if after:
        lines.append("\nUnités les plus ratées après : " + ", ".join(f"{n} ({c})" for n, c in after["most_missed"][:10]))
    (ROOT / "runs/detector/FINETUNE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    log(f"=== réglage fin : fin ({'remplacé' if better else 'gardé l’ancien'}) ===")


if __name__ == "__main__":
    main()
