"""Enchaîne les matchs ET fait progresser le scanner en même temps.

- après chaque match : ré-étiquetage à double vérification (scripts/double_check.py) lancé EN PARALLÈLE du
  match suivant (léger : ~1 image/1,5 s de jeu à relire) ;
- tous les --every matchs : réglage fin du scanner sur les images validées (~20 min, entre deux matchs pour ne
  pas ralentir la détection en jeu), adopté SEULEMENT s'il fait mieux sur le jeu de test « notre écran »
  (matchs jamais appris) sans perdre sur le test habituel ;
- le score du scanner est noté à chaque fois dans runs/detector/scanner_progress.jsonl.

  .venv-yolo\\Scripts\\python scripts/play_and_learn.py --games 20 --every 4
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("D:/clash-ai-dataset")
MODELS = ROOT / "models/yolo"
PY = str(ROOT / ".venv-yolo/Scripts/python.exe")
PY_LIGHT = str(ROOT / ".venv/Scripts/python.exe")
FT = DATA / "dc_ft"
ENV = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def screen_score(weights: Path, imgsz: int) -> dict:
    """F1 (nom ET camp justes, IoU >= 0.5) sur le jeu de test « notre écran » (matchs jamais appris)."""
    from ultralytics import YOLO
    names = yaml.safe_load(open(DATA / "data_v2.yaml"))["names"]
    m = YOLO(str(weights), task="detect")
    tp = fp = fn = 0
    for img in sorted((DATA / "images/dc_test").glob("*.jpg")):
        W, H = 568, 896
        gt = []
        for l in (DATA / "labels/dc_test" / f"{img.stem}.txt").read_text().splitlines():
            c, x, y, w, h = l.split()
            x, y, w, h = float(x) * W, float(y) * H, float(w) * W, float(h) * H
            gt.append((names[int(c)], (x - w / 2, y - h / 2, x + w / 2, y + h / 2)))
        r = m.predict(str(img), imgsz=imgsz, conf=0.35, verbose=False, device=0)[0]
        used = set()
        for (b, c) in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist()):
            j = next((k for k, (n, g) in enumerate(gt) if k not in used and n == r.names[int(c)] and iou(b, g) >= 0.5), None)
            if j is None:
                fp += 1
            else:
                used.add(j)
                tp += 1
        fn += len(gt) - len(used)
    p, rc = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return {"f1": round(2 * p * rc / max(1e-9, p + rc), 3), "precision": round(p, 3), "recall": round(rc, 3),
            "boxes": tp + fn}


def katacr_f1(weights: Path, imgsz: int) -> float:
    out = subprocess.run([PY, "scripts/eval_yolo.py", str(weights), f"--imgsz={imgsz}"], cwd=ROOT, env=ENV,
                         capture_output=True, text=True, encoding="utf-8").stdout
    return json.loads(out[out.find("{"):])["f1"]


def finetune(minutes: float) -> None:
    imgsz = json.loads((MODELS / "clashai_yolo11s.json").read_text())["imgsz"]
    mine = sorted((DATA / "images/train_fk").glob("dc_*.jpg"))
    if len(list((DATA / "images/dc_test").glob("*.jpg"))) < 10 or len(mine) < 40:
        print(f"[scanner] pas encore assez d'images validées ({len(mine)}) : réglage fin plus tard", flush=True)
        return
    shutil.rmtree(FT, ignore_errors=True)
    for sub in ("images/train", "labels/train"):
        (FT / sub).mkdir(parents=True)
    random.seed(len(mine))
    # nos images validées x3 + un échantillon des images habituelles (pour ne rien oublier du reste)
    pool, fk = sorted((DATA / "images/train").glob("*.jpg")), sorted((DATA / "images/train_fk").glob("fk_*.jpg"))
    others = random.sample(pool, min(3000, len(pool))) + random.sample(fk, min(1500, len(fk)))
    for rep in range(3):
        for img in mine:
            shutil.copy(img, FT / "images/train" / f"r{rep}_{img.name}")
            shutil.copy(DATA / "labels/train_fk" / f"{img.stem}.txt", FT / "labels/train" / f"r{rep}_{img.stem}.txt")
    for img in others:
        lab = img.parent.parent.parent / "labels" / img.parent.name / f"{img.stem}.txt"
        shutil.copy(img, FT / "images/train" / img.name)
        if lab.exists():
            shutil.copy(lab, FT / "labels/train" / f"{img.stem}.txt")
    d = yaml.safe_load(open(DATA / "data_v2.yaml"))
    d.update(path=str(FT), train="images/train", val=str(DATA / "images/val"))
    yaml.safe_dump(d, open(FT / "data.yaml", "w"), sort_keys=False)
    shutil.rmtree(ROOT / "runs/detector/dc_ft", ignore_errors=True)
    subprocess.run([PY, "scripts/train_yolo.py", "--model", str(MODELS / "clashai_yolo11s.pt"), "--data", str(FT / "data.yaml"),
                    "--hours", str(minutes / 60), "--lr0", "0.002", "--close-mosaic", "0", "--workers", "4",
                    "--name", "dc_ft", "--imgsz", str(imgsz), "--batch", "8"], cwd=ROOT, env=ENV)
    best = ROOT / "runs/detector/dc_ft/weights/best.pt"
    if not best.exists():
        print("[scanner] réglage fin sans résultat", flush=True)
        return
    subprocess.run([PY, "-c", f"from ultralytics import YOLO; YOLO(r'{best}').export(format='engine', half=True, imgsz={imgsz}, device=0)"],
                   cwd=ROOT, env=ENV)
    new = best.with_suffix(".engine")
    before, after = screen_score(MODELS / "clashai_yolo11s.engine", imgsz), screen_score(new, imgsz)
    kb, ka = katacr_f1(MODELS / "clashai_yolo11s.engine", imgsz), katacr_f1(new, imgsz)
    # le test « notre écran » ne contient que ce que le scanner voyait déjà (étiquettes par accord) : il vérifie
    # qu'on ne régresse pas. Le PROGRÈS se mesure sur nos troupes posées (matchs de test) : part retrouvée
    sys.path.insert(0, str(ROOT / "scripts"))
    from quick_finetune import is_test_game, own_recall
    own = [p.stem for p in (DATA / "labels/train_fk").glob("own_*.txt") if is_test_game(p.stem)]
    ob, oa = own_recall(MODELS / "clashai_yolo11s.engine", imgsz, own), own_recall(new, imgsz, own)
    ok = oa > ob and after["f1"] >= before["f1"] - 0.01 and ka >= kb - 0.005
    rec = {"t": time.strftime("%Y-%m-%d %H:%M"), "images": len(mine), "screen_before": before, "screen_after": after,
           "own_before": round(ob, 3), "own_after": round(oa, 3), "katacr_before": kb, "katacr_after": ka, "adopted": ok}
    with open(ROOT / "runs/detector/scanner_progress.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"[scanner] nos troupes vues {ob:.0%} -> {oa:.0%} | notre écran F1 {before['f1']:.3f} -> {after['f1']:.3f} | test habituel {kb:.3f} -> {ka:.3f} : "
          f"{'ADOPTÉ' if ok else 'pas adopté'}", flush=True)
    if ok:
        for ext in (".pt", ".engine"):
            shutil.copy(MODELS / f"clashai_yolo11s{ext}", MODELS / f"clashai_yolo11s_prev{ext}")
        # copie à côté puis remplacement : jamais un .pt neuf avec l'ancien .engine si une copie échoue en route
        for src, ext in ((new, ".engine"), (best, ".pt")):
            tmp = MODELS / f"clashai_yolo11s.tmp{ext}"
            shutil.copy(src, tmp)
        for ext in (".engine", ".pt"):
            os.replace(MODELS / f"clashai_yolo11s.tmp{ext}", MODELS / f"clashai_yolo11s{ext}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=0, help="0 = sans fin")
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--no-show", action="store_true")
    a = ap.parse_args()
    import threading
    labeler, trainer = None, None
    g = 0
    while a.games == 0 or g < a.games:
        g += 1
        if labeler is None or labeler.poll() is not None:
            # relit en parallèle les matchs pas encore traités pendant qu'on joue le suivant
            labeler = subprocess.Popen([PY, "scripts/double_check.py"], cwd=ROOT, env=ENV,
                                       creationflags=getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0))
        print(f"=== match {g}/{a.games or '∞'} ===", flush=True)
        r = subprocess.run([PY, "scripts/autoplay.py", "--games", "1"] + ([] if a.no_show else ["--show"]),
                           cwd=ROOT, env=ENV)
        subprocess.run([PY_LIGHT, "scripts/review_game.py"], cwd=ROOT, env=ENV)
        if r.returncode != 0:
            print("autoplay en échec : arrêt (téléphone débranché ?)", flush=True)
            break
        if g % a.every == 0 and (trainer is None or not trainer.is_alive()):
            # réglage fin EN PARALLÈLE des matchs (le téléphone ne reste jamais sans jouer) ; le nouveau scanner,
            # s'il est adopté, est chargé au match suivant
            def job():
                subprocess.run([PY, "scripts/double_check.py"], cwd=ROOT, env=ENV)
                subprocess.run([PY, "scripts/own_labels.py"], cwd=ROOT, env=ENV)   # test « nos troupes » à jour
                finetune(a.minutes)
            trainer = threading.Thread(target=job, daemon=False)
            trainer.start()
    if labeler:
        labeler.wait()
    if trainer:
        trainer.join()


if __name__ == "__main__":
    main()
