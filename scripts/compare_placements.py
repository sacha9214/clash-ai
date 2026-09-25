"""Compare où les joueurs des vidéos ont posé leurs cartes et où NOTRE cerveau les aurait posées.

Pour chaque carte de notre deck jouée dans une vidéo (runs/videos/*.jsonl) :
- on reconstruit le plateau du point de vue de celui qui joue (retourné si c'est le joueur du haut),
  sans l'unité qu'il vient de poser ;
- on demande au cerveau où il poserait cette carte (seule en main, élixir plein pour ne pas bloquer) ;
- on mesure l'écart en cases, le même couloir ou non, et s'il aurait attendu.
Les coups des gagnants sont comptés à part.

  .venv-katacr\\Scripts\\python scripts/compare_placements.py
"""
from __future__ import annotations

import collections
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from clashai.brain import RIVER_Y, Brain, Seen  # noqa: E402
from clashai.cards import DECK  # noqa: E402
from clashai.opponent import UNIT2CARD  # noqa: E402
from clashai.tiles import PHONE, ROWS  # noqa: E402


def to_phone(x: float, y: float, grid: dict, flip: bool) -> tuple[float, float]:
    """Arène de la vidéo -> écran du téléphone, via les cases (retourné si le joueur est en haut)."""
    tx, ty = (x - grid["x0"]) / grid["tw"], (y - grid["y0"]) / grid["th"]
    if flip:
        tx, ty = 18 - tx, ROWS - ty            # demi-tour : son camp devient le bas de l'écran
    return PHONE.to_frac(tx, ty)


def main():
    rows = []
    for f in sorted(glob.glob(str(ROOT / "runs/videos/*.jsonl"))):
        if f.endswith(".raw.jsonl") or not Path(f.replace(".jsonl", ".battles.json")).exists():
            continue
        vid = Path(f).stem
        meta = json.load(open(ROOT / f"runs/videos/{vid}.battles.json", encoding="utf-8"))
        grid = meta["grid"]
        for line in open(f, encoding="utf-8"):
            e = json.loads(line)
            if e["card"] not in DECK:
                continue
            flip = e["side"] == "opponent"
            units = DECK[e["card"]].units
            board, removed = [], False
            for name, enemy, x, y in e["board"]:
                if not removed and name in units and abs(x - e["x"]) < 0.03 and abs(y - e["y"]) < 0.03:
                    removed = True             # l'unité qu'il vient de poser : pas encore là au moment de décider
                    continue
                px, py = to_phone(x, y, grid, flip)
                # camp : celui du détecteur, retourné avec le plateau
                seen_enemy = bool(enemy) != flip
                if name in UNIT2CARD:              # comme Brain.to_seen : unités seulement, pas les tours
                    board.append(Seen(name, seen_enemy, px, py))
            # situation : l'ennemi le plus avancé sur notre moitié (défense), sinon attaque / temps calme
            near = [b for b in board if b.enemy and b.y > RIVER_Y - 0.02]
            threat = max(near, key=lambda b: b.y) if near else None
            context = f"défense : {threat.name}" if threat else "attaque / calme"
            brain = Brain({"ignore_small": True})
            d = brain.decide(board, [e["card"], None, None, None], [True, False, False, False], 10.0, e["t"])
            pro = PHONE.cell(*to_phone(e["x"], e["y"], grid, flip))
            rec = {"video": vid, "t": e["t"], "card": e["card"], "winner": e.get("winner") == e["side"],
                   "context": context, "threat_lane": (0 if threat.x < 0.5 else 1) if threat else None,
                   "pro": pro, "ai": d.tile if d else None, "reason": d.reason if d else "attendrait"}
            if d:
                rec["dist"] = float(np.hypot(d.tile[0] - pro[0], d.tile[1] - pro[1]))
                rec["same_lane"] = (d.tile[0] < 9) == (pro[0] < 9)
            rows.append(rec)
    out = ROOT / "runs/videos/placement_compare.jsonl"
    with open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"{len(rows)} coups de pros avec une carte de notre deck\n")
    print(f"{'carte':11s} {'coups':>5s} {'IA joue':>8s} {'écart méd.':>10s} {'même couloir':>13s}   (gagnants : écart)")
    by = collections.defaultdict(list)
    for r in rows:
        by[r["card"]].append(r)
    for card, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        played = [r for r in rs if r["ai"]]
        win = [r["dist"] for r in played if r["winner"]]
        med = np.median([r["dist"] for r in played]) if played else float("nan")
        lane = np.mean([r["same_lane"] for r in played]) if played else float("nan")
        wins = f"({np.median(win):.1f}c sur {len(win)})" if win else ""
        print(f"{card:11s} {len(rs):5d} {len(played) / len(rs):8.0%} {med:9.1f}c {lane:12.0%}   {wins}")
    print("\nPlus gros désaccords (coups de gagnants) :")
    worst = sorted([r for r in rows if r["ai"] and r["winner"]], key=lambda r: -r["dist"])[:12]
    for r in worst:
        print(f"  {r['video']} {r['t']:6.1f}s {r['card']:10s} pro {tuple(r['pro'])} / IA {tuple(r['ai'])} "
              f"({r['dist']:.0f} cases) — {r['reason']}")
    waits = collections.Counter(r["card"] for r in rows if not r["ai"])
    print("\nL'IA aurait attendu au lieu de jouer :", dict(waits))


if __name__ == "__main__":
    main()
