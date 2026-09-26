"""Pré-génère des images synthétiques KataCR au format YOLO standard (pour un modèle récent, ex. YOLO11).

KataCR colle des sprites découpés d'unités sur des arènes vides et connaît donc la boîte exacte de
chaque unité. Leur entraînement génère ces images à la volée ; ici on les écrit sur disque une fois
pour pouvoir utiliser les outils ultralytics actuels (export TensorRT, etc.).

Le camp fait partie de la classe : « knight_0 » (le nôtre, en bas) / « knight_1 » (ennemi), ce qui
permet un détecteur standard sans la sortie « camp » maison de KataCR.
Validation : les images réelles annotées à la main de KataCR, converties au même format.

  .venv-katacr\\Scripts\\python scripts/gen_synthetic.py --n 60000 --out D:/clash-ai-dataset
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KATACR = ROOT / "third_party/KataCR"
sys.path.insert(0, str(KATACR))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402


def class_names() -> list[str]:
    """Union des classes des deux détecteurs KataCR (sans l'interface), chacune en 2 camps."""
    names = set()
    for d in ("detector1", "detector2"):
        data = yaml.safe_load(open(KATACR / f"katacr/yolov8/{d}/data.yaml", encoding="utf-8"))
        names |= set(data["names"].values())
    from clashai.detect import UI
    units = sorted(n for n in names if n not in UI and n not in ("text", "selected"))
    return [f"{n}_{bel}" for n in units for bel in (0, 1)]


def _worker(args):
    wid, start, count, out, names, only, prefix = args
    os.chdir(KATACR)
    from katacr.build_dataset.generator import Generator
    from katacr.constants.label_list import idx2unit
    from katacr.yolov8.cfg import img_size, intersect_ratio_thre, map_update_mode, unit_nums
    units = sorted(only) if only else sorted({n.rsplit("_", 1)[0] for n in names})   # --only : images centrées sur ces unités
    cls_id = {n: i for i, n in enumerate(names)}
    gen = Generator(seed=1000 + wid, intersect_ratio_thre=intersect_ratio_thre,
                    map_update={"mode": map_update_mode, "size": 5}, avail_names=units, noise_unit_ratio=0)
    for k in range(start, start + count):
        gen.reset()
        gen.add_tower()
        gen.add_unit(unit_nums)
        img, box, _ = gen.build(box_format="cxcywh", img_size=img_size)
        lines = []
        for cx, cy, w, h, bel, c in box:
            key = f"{idx2unit[int(c)]}_{int(bel)}"
            if key in cls_id:
                lines.append(f"{cls_id[key]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        stem = f"{prefix}_{k:06d}"
        cv2.imwrite(str(out / "images/train" / f"{stem}.jpg"), img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
        (out / "labels/train" / f"{stem}.txt").write_text("\n".join(lines))
    return count


def convert_val(out: Path, names: list[str]) -> int:
    """Images réelles annotées de KataCR -> même format (validation)."""
    sys.path.insert(0, str(ROOT))
    from katacr.constants.label_list import idx2unit
    cls_id = {n: i for i, n in enumerate(names)}
    src = ROOT / "data/Clash-Royale-Dataset/images/part2"
    n = 0
    for line in open(src / "yolo_annotation.txt"):
        f = src / line.strip()
        if not line.strip() or not f.exists():
            continue
        lines = []
        for lab in open(f.with_suffix(".txt")):
            p = lab.split()
            key = f"{idx2unit[int(p[0])]}_{int(p[5])}"
            if key in cls_id:
                lines.append(f"{cls_id[key]} {' '.join(p[1:5])}")
        stem = f"val_{n:05d}"
        img = cv2.imread(str(f))
        cv2.imwrite(str(out / "images/val" / f"{stem}.jpg"), img)
        (out / "labels/val" / f"{stem}.txt").write_text("\n".join(lines))
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60000)
    ap.add_argument("--out", default="D:/clash-ai-dataset")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--skip-val", action="store_true")
    ap.add_argument("--only", default="", help="unités à mettre en avant, séparées par des virgules (points faibles)")
    ap.add_argument("--prefix", default="syn")
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT))
    out = Path(a.out)
    for sub in ("images/train", "labels/train", "images/val", "labels/val"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    names = class_names()
    if not (out / "data.yaml").exists():
        yaml.safe_dump({"path": str(out), "train": "images/train", "val": "images/val",
                        "names": dict(enumerate(names))}, open(out / "data.yaml", "w"), sort_keys=False)
    print(f"{len(names)} classes ({len(names) // 2} unités x 2 camps)", flush=True)
    if not a.skip_val:
        print("validation réelle :", convert_val(out, names), "images", flush=True)
    per = -(-a.n // a.workers)
    only = [u for u in a.only.split(",") if u]
    jobs = [(w, w * per, min(per, a.n - w * per), out, names, only, a.prefix) for w in range(a.workers) if w * per < a.n]
    with mp.Pool(len(jobs)) as pool:
        done = sum(pool.map(_worker, jobs))
    print("images synthétiques :", done, flush=True)


if __name__ == "__main__":
    main()
