"""Entraînement « v2 » du détecteur : les 300 classes actuelles + nouvelles cartes, héros et évolutions.

Autonome (tourne même si Claude est fermé) :
1. attend la fin des téléchargements Fan Kit / du réglage fin en cours (--wait-pid) ;
2. fabrique les images Fan Kit (fankit_synth.py) ;
3. ajoute les images réelles corrigées dans label_tool.py (numéros de classes convertis vers data_v2.yaml) ;
4. entraîne depuis le détecteur actuel (les couches communes sont reprises, la dernière est agrandie) ;
5. note sur le jeu de test (cartes connues) : remplace le détecteur de l'IA seulement s'il reste au moins aussi bon.
Les nouvelles cartes n'ont pas encore de vérité terrain réelle : on rapporte combien le modèle en voit sur les
vidéos récentes (à vérifier à l'œil).

  .venv-yolo\\Scripts\\python scripts/train_v2.py --hours 8 --wait-pid 111 222
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
import night_detector  # noqa: E402
from night_detector import PY_YOLO, last_eval, log, run  # noqa: E402

night_detector.LOG = ROOT / "runs/detector/train_v2.log"
DATA = Path("D:/clash-ai-dataset")
MODELS = ROOT / "models/yolo"
NAME = "yolo11s_cr_v2"


def pid_alive(pid: int) -> bool:
    return str(pid) in subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True).stdout


def add_real_labels() -> int:
    """Images corrigées à la main -> train_fk, avec les numéros de classes de data_v2.yaml."""
    import yaml
    real = DATA / "real"
    if not (real / "labels").exists():
        return 0
    old = json.loads((real / "classes.json").read_text(encoding="utf-8")) if (real / "classes.json").exists() else \
        [v for _, v in sorted(yaml.safe_load(open(DATA / "data.yaml"))["names"].items())]
    v2 = yaml.safe_load(open(DATA / "data_v2.yaml"))["names"]
    new_id = {n: i for i, n in v2.items()}
    n = 0
    for lab in sorted((real / "labels").glob("*.txt")):
        img = real / "images" / f"{lab.stem}.jpg"
        if not img.exists():
            continue
        lines = []
        for l in lab.read_text().splitlines():
            if l.strip():
                c, *box = l.split()
                name = old[int(c)] if int(c) < len(old) else None
                if name in new_id:
                    lines.append(" ".join([str(new_id[name]), *box]))
        # chaque image réelle compte 3 fois : peu nombreuses mais précieuses (vrai rendu du jeu)
        for rep in range(3):
            shutil.copy(img, DATA / "images/train_fk" / f"real{rep}_{lab.stem}.jpg")
            (DATA / "labels/train_fk" / f"real{rep}_{lab.stem}.txt").write_text("\n".join(lines))
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--n-fk", type=int, default=25000)
    ap.add_argument("--wait-pid", type=int, nargs="*", default=[])
    a = ap.parse_args()
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    hold = ROOT / "runs/HOLD"                 # matchs sur le téléphone en cours : le GPU doit rester libre
    if hold.exists():
        log("en attente : runs/HOLD existe (matchs sur le téléphone)")
        while hold.exists():
            time.sleep(30)
    for pid in a.wait_pid:
        log(f"attente de la fin du processus {pid}…")
        while pid_alive(pid):
            time.sleep(60)
    log("=== entraînement v2 : début ===")
    run([PY_YOLO, "scripts/fankit_synth.py", "--n", a.n_fk])
    log(f"images réelles corrigées ajoutées : {add_real_labels()}")

    code, out = run([PY_YOLO, "scripts/eval_yolo.py", MODELS / "clashai_yolo11s.engine"])
    before = last_eval(out)
    run([PY_YOLO, "scripts/train_yolo.py", "--model", MODELS / "clashai_yolo11s.pt", "--data", DATA / "data_v2.yaml",
         "--hours", a.hours, "--lr0", "0.005", "--close-mosaic", "3", "--workers", "11", "--name", NAME])
    found = sorted(ROOT.glob(f"runs/**/{NAME}/weights/best.pt"), key=lambda p: p.stat().st_mtime)
    if not found:
        log("pas de modèle : voir le journal")
        return
    best = found[-1]
    run([PY_YOLO, "-c", f"from ultralytics import YOLO; YOLO(r'{best}').export(format='engine', half=True, imgsz=896, device=0)"])
    engine = best.with_suffix(".engine")
    code, out = run([PY_YOLO, "scripts/eval_yolo.py", engine if engine.exists() else best])
    after = last_eval(out)
    ok = bool(before and after and after["f1"] >= before["f1"] - 0.005)   # gagner des cartes sans rien perdre
    if ok and engine.exists():
        for ext in (".pt", ".engine"):
            shutil.copy(MODELS / f"clashai_yolo11s{ext}", MODELS / f"clashai_yolo11s_before_v2{ext}")
        shutil.copy(best, MODELS / "clashai_yolo11s.pt")
        shutil.copy(engine, MODELS / "clashai_yolo11s.engine")
    lines = ["# Détecteur v2 (nouvelles cartes, héros, évolutions)", "",
             "| Modèle | Précision | Rappel | F1 (cartes connues) | Camp | ms/image |", "|---|---|---|---|---|---|"]
    for n, r in (("Avant", before), ("v2", after)):
        lines.append(f"| {n} | {r['precision']:.1%} | {r['recall']:.1%} | **{r['f1']:.3f}** | {r['side_accuracy']:.1%} | {r['ms_per_image']} |"
                     if r else f"| {n} | — | — | — | — | — |")
    lines += ["", f"**{'Adopté : l’IA reconnaît maintenant les nouvelles cartes' if ok else 'Pas adopté : il perd en précision sur les cartes connues'}.**",
              "", "Nouvelles cartes : pas encore de test réel (il faut des images étiquetées à la main) — à vérifier à l'œil."]
    (ROOT / "runs/detector/V2_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    log(f"=== entraînement v2 : fin ({'adopté' if ok else 'non adopté'}) ===")


if __name__ == "__main__":
    main()
