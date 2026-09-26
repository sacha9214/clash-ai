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
import os
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
IMGSZ = 1024


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


def train_with_watchdog(hours: float, start_from: str | None) -> None:
    """Entraîne en surveillant : sous Windows, les processus qui chargent les images peuvent se bloquer
    (le GPU affiche 100 % mais plus rien n'avance). Si last.pt n'est pas réécrit pendant STALL_MIN minutes,
    on arrête et on repart de last.pt avec le temps restant."""
    import subprocess
    STALL_MIN = 75                              # une époque ~45 min à 1024 px + validation
    t_end = time.time() + hours * 3600
    weights = Path(start_from) if start_from else MODELS / "clashai_yolo11s.pt"
    last = ROOT / f"runs/detector/{NAME}/weights/last.pt"
    attempt = 0
    while time.time() < t_end - 1800:
        attempt += 1
        left = round((t_end - time.time()) / 3600, 2)
        cmd = [str(PY_YOLO), "scripts/train_yolo.py", "--model", str(weights), "--data", str(DATA / "data_v2.yaml"),
               "--hours", str(left), "--lr0", "0.005" if attempt == 1 and not start_from else "0.003",
               "--close-mosaic", "3", "--workers", "8", "--name", NAME, "--imgsz", str(IMGSZ), "--batch", "12"]
        log(f"essai {attempt} : {left} h, depuis {weights.name}")
        p = subprocess.Popen(cmd, cwd=ROOT, env=dict(os.environ, PYTHONIOENCODING="utf-8"),
                             stdout=open(ROOT / f"runs/detector/train_v2_try{attempt}.txt", "w"), stderr=subprocess.STDOUT)
        t_start = time.time()
        while p.poll() is None:
            time.sleep(60)
            ref = last.stat().st_mtime if last.exists() and last.stat().st_mtime > t_start else t_start
            if time.time() - ref > STALL_MIN * 60:
                log(f"bloqué depuis {STALL_MIN} min : arrêt et reprise depuis last.pt")
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)
                break
        if p.poll() == 0:
            log("entraînement terminé normalement")
            return
        if last.exists():
            keep = ROOT / f"runs/detector/{NAME}_resume.pt"      # le prochain essai réécrit le dossier
            keep.write_bytes(last.read_bytes())
            best = last.with_name("best.pt")
            if best.exists():
                best.with_name("best_before_resume.pt").write_bytes(best.read_bytes())
            weights = keep
    log("temps écoulé")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--n-fk", type=int, default=25000)
    ap.add_argument("--wait-pid", type=int, nargs="*", default=[])
    ap.add_argument("--start-from", default=None, help="reprendre depuis ces poids (ex. last.pt d'un essai bloqué)")
    ap.add_argument("--skip-prep", action="store_true", help="images déjà prêtes : passer directement à l'entraînement")
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
    have = a.n_fk if a.skip_prep else None
    have = have if have is not None else len(list((DATA / "images/train_fk").glob("fk_*.jpg"))) if (DATA / "images/train_fk").exists() else 0
    if have >= a.n_fk:
        log(f"images Fan Kit déjà prêtes : {have}")         # générées pendant que le GPU faisait autre chose
    else:
        run([PY_YOLO, "scripts/fankit_synth.py", "--n", a.n_fk])
    log(f"images réelles corrigées ajoutées : {add_real_labels()}")

    code, out = run([PY_YOLO, "scripts/eval_yolo.py", MODELS / "clashai_yolo11s.engine"])
    before = last_eval(out)
    train_with_watchdog(a.hours, a.start_from)
    found = sorted(ROOT.glob(f"runs/**/{NAME}/weights/best.pt"), key=lambda p: p.stat().st_mtime)
    if not found:
        log("pas de modèle : voir le journal")
        return
    best = found[-1]
    run([PY_YOLO, "-c", f"from ultralytics import YOLO; YOLO(r'{best}').export(format='engine', half=True, imgsz={IMGSZ}, device=0)"])
    engine = best.with_suffix(".engine")
    code, out = run([PY_YOLO, "scripts/eval_yolo.py", engine if engine.exists() else best, f"--imgsz={IMGSZ}"])
    after = last_eval(out)
    ok = bool(before and after and after["f1"] >= before["f1"] - 0.005)   # gagner des cartes sans rien perdre
    if ok and engine.exists():
        for ext in (".pt", ".engine"):
            shutil.copy(MODELS / f"clashai_yolo11s{ext}", MODELS / f"clashai_yolo11s_before_v2{ext}")
        shutil.copy(best, MODELS / "clashai_yolo11s.pt")
        shutil.copy(engine, MODELS / "clashai_yolo11s.engine")
        (MODELS / "clashai_yolo11s.json").write_text(json.dumps({"imgsz": IMGSZ}))   # l'IA lit la taille ici
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
