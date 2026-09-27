"""Rassemble dans learning/samples/ ce qu'il faut pour améliorer la détection, l'horloge et l'imitation,
pour l'envoyer sur GitHub en une commande :
  - N images brutes tirées au hasard dans runs/capture/**.jpg, réparties entre les matchs, réduites (JPEG ~85) ;
  - les petits fichiers de coups des vidéos : runs/videos/*.jsonl et *.battles.json (pas les *.raw.jsonl, rien > 5 Mo) ;
  - manifest.json : nombres, tailles, chemins d'origine.
Plafond ~40 Mo au total : on met moins d'images plutôt que de le dépasser. frames/ et videos/ sont refaits à chaque fois.

  python scripts/pack_samples.py                  # 150 images, côté long 960 px
  python scripts/pack_samples.py --frames 300 --side 1280 --seed 1
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CAPTURE = ROOT / "runs/capture"
VIDEOS = ROOT / "runs/videos"
OUT = ROOT / "learning/samples"
MAX_MB = 40
MAX_FILE = 5 * 1024 * 1024          # fichier vidéo plus gros : ignoré (les *.raw.jsonl font des centaines de Mo)
MANIFEST_ROOM = 256 * 1024          # place gardée pour manifest.json


def _mb(n: int) -> str:
    return f"{n / 1024 / 1024:.1f} Mo"


def _rel(p: Path) -> str:
    """Chemin d'origine lisible (relatif au dépôt si possible, toujours avec des /)."""
    try:
        return p.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return p.resolve().as_posix()


def spread(files: list[Path], rng: random.Random) -> list[Path]:
    """Toutes les images dans un ordre qui alterne les matchs (dossiers), au hasard dans chaque match :
    en prendre les n premières donne n images réparties sur tous les matchs."""
    groups: dict[Path, list[Path]] = {}
    for f in sorted(files):
        groups.setdefault(f.parent, []).append(f)
    order = list(groups.values())
    for g in order:
        rng.shuffle(g)
    rng.shuffle(order)
    out = []
    while any(order):
        out += [g.pop() for g in order if g]
    return out


def _encode(path: Path, side: int, quality: int) -> bytes | None:
    """Image réduite (côté long <= side) en JPEG ; lecture par octets : chemins Windows non ASCII compris."""
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    if max(h, w) > side:
        s = side / max(h, w)
        img = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else None


def pack(frames: int = 150, side: int = 960, quality: int = 85, max_mb: float = MAX_MB, seed: int | None = None) -> dict:
    rng = random.Random(seed)
    budget = int(max_mb * 1024 * 1024) - MANIFEST_ROOM
    vids = sorted((p for p in [*VIDEOS.glob("*.jsonl"), *VIDEOS.glob("*.battles.json")]
                   if not p.name.endswith(".raw.jsonl")), key=lambda p: p.stat().st_size) if VIDEOS.is_dir() else []
    found = sorted(CAPTURE.rglob("*.jpg")) if CAPTURE.is_dir() else []
    if not vids and not found:   # rien à prendre : l'envoi précédent reste tel quel
        return {"frames": {"count": 0, "asked": frames, "available": 0, "games": 0, "bytes": 0, "capped": False},
                "videos": {"count": 0, "bytes": 0, "skipped": []}, "total_bytes": 0, "files": []}
    for sub in ("frames", "videos"):
        shutil.rmtree(OUT / sub, ignore_errors=True)
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    files, skipped, used = [], [], 0
    # 1) fichiers de coups des vidéos (petits, tous utiles à l'imitation) : les plus petits d'abord
    for p in vids:
        size = p.stat().st_size
        if size > MAX_FILE or used + size > budget:
            skipped.append({"source": _rel(p), "bytes": size, "why": "> 5 Mo" if size > MAX_FILE else "plafond"})
            continue
        shutil.copyfile(p, OUT / "videos" / p.name)
        files.append({"path": f"videos/{p.name}", "source": _rel(p), "bytes": size})
        used += size
    # 2) images brutes, réparties entre les matchs, jusqu'à N ou jusqu'au plafond
    n, capped, games = 0, False, set()
    for p in spread(found, rng):
        if n >= frames:
            break
        data = _encode(p, side, quality)
        if data is None:
            continue
        if used + len(data) > budget:
            capped = True
            break
        rel = p.relative_to(CAPTURE)
        name = "_".join(rel.parts)   # <match>_<image>.jpg : un seul dossier, noms uniques
        (OUT / "frames" / name).write_bytes(data)
        files.append({"path": f"frames/{name}", "source": _rel(p), "bytes": len(data)})
        games.add(rel.parts[0] if len(rel.parts) > 1 else "")
        used += len(data)
        n += 1
    fb = sum(f["bytes"] for f in files if f["path"].startswith("frames/"))
    manifest = {"created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "frames": {"count": n, "asked": frames, "available": len(found), "games": len(games), "bytes": fb,
                           "side": side, "quality": quality, "capped": capped},
                "videos": {"count": len(files) - n, "bytes": used - fb, "skipped": skipped},
                "total_bytes": used, "max_bytes": int(max_mb * 1024 * 1024), "files": files}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    manifest["total_bytes"] = used + (OUT / "manifest.json").stat().st_size
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--frames", type=int, default=150, help="nombre d'images brutes (réparties entre les matchs)")
    ap.add_argument("--side", type=int, default=960, help="côté long maximal des images, en pixels")
    ap.add_argument("--quality", type=int, default=85, help="qualité JPEG")
    ap.add_argument("--max-mb", type=float, default=MAX_MB, help="plafond total (Mo) : moins d'images au-delà")
    ap.add_argument("--seed", type=int, help="tirage reproductible")
    a = ap.parse_args(argv)
    m = pack(a.frames, a.side, a.quality, a.max_mb, a.seed)
    f, v = m["frames"], m["videos"]
    print(f"{_rel(OUT)} : {f['count']} images ({_mb(f['bytes'])}, {f['games']} matchs, {f['available']} disponibles)"
          f" + {v['count']} fichiers vidéo ({_mb(v['bytes'])}) = {_mb(m['total_bytes'])}")
    if f["capped"]:
        print(f"plafond de {a.max_mb:g} Mo atteint : {f['count']} images au lieu de {f['asked']}")
    for s in v["skipped"]:
        print(f"ignoré : {s['source']} ({_mb(s['bytes'])}, {s['why']})")
    if not m["files"]:
        print(f"rien à envoyer : ni {_rel(CAPTURE)}/**.jpg ni {_rel(VIDEOS)}/*.jsonl")
        return m
    print("Pour l'envoyer :")
    if Path.cwd().resolve() != ROOT.resolve():
        print(f'  cd "{ROOT}"')
    print(f"  git add {_rel(OUT)}")
    print(f'  git commit -m "Échantillons : {f["count"]} images, {v["count"]} fichiers vidéo"')
    print("  git push")
    return m


if __name__ == "__main__":
    main()
