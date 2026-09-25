"""Nuit d'entraînement du détecteur, autonome (tourne même si Claude est fermé).

1. Note de KataCR sur le jeu de test (séquences jamais vues) — la référence à battre.
2. Entraînement YOLO11 (train_yolo.py) ; en cas d'échec (mémoire GPU…), nouvel essai avec un lot plus petit.
3. Note du meilleur modèle, export TensorRT (FP16), note de la version TensorRT (vitesse réelle).
4. Rapport lisible : runs/detector/NIGHT_REPORT.md (+ journal : runs/detector/night.log).

  .venv-yolo\\Scripts\\python scripts/night_detector.py --hours 8
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
LOG = ROOT / "runs/detector/night.log"
REPORT = ROOT / "runs/detector/NIGHT_REPORT.md"
PY_YOLO = ROOT / ".venv-yolo/Scripts/python.exe"
PY_KATACR = ROOT / ".venv-katacr/Scripts/python.exe"
NAME = "yolo11s_cr"


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd: list, env_extra: dict | None = None) -> tuple[int, str]:
    env = dict(os.environ, PYTHONIOENCODING="utf-8", **(env_extra or {}))
    log("$ " + " ".join(str(c) for c in cmd))
    p = subprocess.run([str(c) for c in cmd], cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(p.stdout[-20000:] + "\n" + p.stderr[-20000:] + "\n")
    return p.returncode, p.stdout


def last_eval(stdout: str) -> dict | None:
    """Le JSON imprimé à la fin par eval_detector.py / eval_yolo.py."""
    i = stdout.find("{")
    try:
        return json.loads(stdout[i:]) if i >= 0 else None
    except json.JSONDecodeError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) - 1),
                    help="processus de préparation des images (le GPU attend s'il y en a trop peu)")
    a = ap.parse_args()
    LOG.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":   # pas de mise en veille pendant toute la nuit (relâché à la fin du programme)
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    log("=== nuit du détecteur : début ===")
    res = {}

    code, out = run([PY_KATACR, "scripts/eval_detector.py"], {"PYTHONPATH": str(ROOT / "third_party/KataCR")})
    res["katacr"] = last_eval(out)
    log(f"KataCR : {res['katacr'] and {k: res['katacr'][k] for k in ('precision', 'recall', 'f1', 'ms_per_image')}}")

    t0, best = time.time(), ROOT / f"runs/detector/{NAME}/weights/best.pt"
    for batch in (16, 8):
        hours_left = a.hours - (time.time() - t0) / 3600
        if hours_left < 0.5:
            break
        code, _ = run([PY_YOLO, "scripts/train_yolo.py", "--model", "yolo11s.pt", "--batch", batch, "--workers", a.workers,
                       "--hours", round(hours_left, 2), "--name", NAME])
        log(f"entraînement (lot {batch}) terminé, code {code}")
        if code == 0 and any(ROOT.glob(f"runs/**/{NAME}/weights/best.pt")):
            break
    res["train_hours"] = round((time.time() - t0) / 3600, 2)
    # ultralytics range parfois les résultats ailleurs (runs/detect/...) : on cherche le modèle partout
    found = sorted(ROOT.glob(f"runs/**/{NAME}/weights/best.pt"), key=lambda p: p.stat().st_mtime)
    if found:
        best = found[-1]
        log(f"meilleur modèle : {best}")

    if best.exists():
        code, out = run([PY_YOLO, "scripts/eval_yolo.py", best])
        res["yolo_pt"] = last_eval(out)
        code, _ = run([PY_YOLO, "-c", f"from ultralytics import YOLO; YOLO(r'{best}').export(format='engine', half=True, imgsz=896, device=0)"])
        engine = best.with_suffix(".engine")
        if engine.exists():
            code, out = run([PY_YOLO, "scripts/eval_yolo.py", engine])
            res["yolo_trt"] = last_eval(out)
        else:
            log("export TensorRT : échec (voir journal) — le modèle .pt reste utilisable")
    else:
        log("aucun modèle entraîné : voir le journal")

    write_report(res)
    log("=== nuit du détecteur : fin ===")


def write_report(res: dict) -> None:
    k, y, t = res.get("katacr"), res.get("yolo_pt"), res.get("yolo_trt")
    lines = ["# Nuit du détecteur", "", f"Entraînement : {res.get('train_hours')} h. Test : séquences vidéo entières jamais vues.", ""]
    rows = [("KataCR (actuel, 2 modèles)", k), ("YOLO11 (nouveau, .pt)", y), ("YOLO11 TensorRT (FP16)", t)]
    lines += ["| Modèle | Précision | Rappel | F1 | Camp | ms/image |", "|---|---|---|---|---|---|"]
    for name, r in rows:
        if r:
            lines.append(f"| {name} | {r['precision']:.1%} | {r['recall']:.1%} | **{r['f1']:.3f}** | {r['side_accuracy']:.1%} | {r['ms_per_image']} |")
        else:
            lines.append(f"| {name} | — | — | — | — | — |")
    new = t or y
    lines.append("")
    if k and new:
        verdict = "MEILLEUR" if new["f1"] > k["f1"] else "PAS MEILLEUR"
        lines.append(f"**Verdict : le nouveau détecteur est {verdict} que KataCR** "
                     f"(F1 {new['f1']:.3f} contre {k['f1']:.3f}, {new['ms_per_image']} ms contre {k['ms_per_image']} ms).")
        lines += ["", "Unités les plus ratées (nouveau) : " + ", ".join(f"{n} ({c})" for n, c in new["most_missed"][:8]),
                  "", "Unités les plus ratées (KataCR) : " + ", ".join(f"{n} ({c})" for n, c in k["most_missed"][:8])]
    else:
        lines.append("**Pas de verdict** : une étape a échoué, voir runs/detector/night.log.")
    lines += ["", "Rien n'a été changé dans l'IA : le passage au nouveau détecteur attend le feu vert de Sacha."]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    log(f"rapport : {REPORT}")


if __name__ == "__main__":
    main()
