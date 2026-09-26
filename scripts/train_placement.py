"""Entraîne le modèle de placement (clashai/placement.py) sur les coups des joueurs des vidéos.

- Coups : runs/videos/<id>.jsonl (carte, case, plateau), grille propre à chaque vidéo (<id>.battles.json).
- Vu du côté de celui qui joue (retourné s'il est en haut). Coups des gagnants x2 ; cartes très fréquentes
  (Squelettes…) ramenées pour ne pas écraser les autres.
- Test honnête : 20 % des VIDÉOS mises de côté (jamais vues), et comparaison avec les règles actuelles
  du cerveau sur les mêmes coups (runs/videos/placement_compare.jsonl).

  .venv-yolo\\Scripts\\python scripts/train_placement.py
"""
from __future__ import annotations

import collections
import glob
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from clashai.placement import COLS, MODEL_DIR, OWN_FIRST_ROW, ROWS, SPELLS, build_net, features, legal_mask  # noqa: E402
from clashai.opponent import UNIT2CARD  # noqa: E402

TEST_SHARE = 0.2


def is_test(vid: str) -> bool:
    return int(hashlib.md5(vid.encode()).hexdigest(), 16) % 100 < TEST_SHARE * 100


def load_moves():
    moves = []
    for f in sorted(glob.glob(str(ROOT / "runs/videos/*.jsonl"))):
        meta_f = Path(f.replace(".jsonl", ".battles.json"))
        if f.endswith(".raw.jsonl") or not meta_f.exists():
            continue
        vid = Path(f).stem
        g = json.load(open(meta_f, encoding="utf-8"))["grid"]
        for line in open(f, encoding="utf-8"):
            e = json.loads(line)
            flip = e["side"] == "opponent"

            def tile(x, y):
                tx, ty = (x - g["x0"]) / g["tw"], (y - g["y0"]) / g["th"]
                return (COLS - tx, ROWS - ty) if flip else (tx, ty)
            units, removed = [], False
            for name, enemy, x, y in e["board"]:
                if not removed and name in UNIT2CARD and UNIT2CARD[name][0] == e["card"] \
                        and abs(x - e["x"]) < 0.03 and abs(y - e["y"]) < 0.03:
                    removed = True                  # l'unité qu'il vient de poser
                    continue
                if name in UNIT2CARD:
                    units.append((name, bool(enemy) != flip, *tile(x, y)))
            tx, ty = tile(e["x"], e["y"])
            c, r = int(np.clip(tx, 0, COLS - 1)), int(np.clip(ty, 0, ROWS - 1))
            if e["card"] not in SPELLS:
                r = max(r, OWN_FIRST_ROW)           # une troupe se pose dans sa moitié (±1 case d'erreur de grille)
            moves.append({"vid": vid, "t": e["t"], "card": e["card"], "units": units, "target": (c, r),
                          "win": e.get("winner") == e["side"]})
    return moves


def main():
    torch.manual_seed(0)
    moves = load_moves()
    cards = sorted({m["card"] for m in moves})
    cid = {c: i for i, c in enumerate(cards)}
    train = [m for m in moves if not is_test(m["vid"])]
    test = [m for m in moves if is_test(m["vid"])]
    freq = collections.Counter(m["card"] for m in train)
    print(f"{len(moves)} coups, {len(cards)} cartes ; apprentissage {len(train)} / test {len(test)} "
          f"({len({m['vid'] for m in test})} vidéos jamais vues)", flush=True)

    def tensors(ms):
        X = torch.from_numpy(np.stack([features(m["units"]) for m in ms]))
        C = torch.tensor([cid[m["card"]] for m in ms])
        Y = torch.tensor([m["target"][1] * COLS + m["target"][0] for m in ms])
        M = torch.from_numpy(np.stack([legal_mask(m["card"]) for m in ms]))
        # poids : gagnants x2, cartes rares relevées (1/racine de la fréquence)
        W = torch.tensor([(2.0 if m["win"] else 1.0) / np.sqrt(freq.get(m["card"], 1)) for m in ms], dtype=torch.float32)
        return X, C, Y, M, W / W.mean()

    Xtr, Ctr, Ytr, Mtr, Wtr = tensors(train)
    net = build_net(len(cards))
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    for epoch in range(60):
        perm = torch.randperm(len(Xtr))
        tot = 0.0
        for i in range(0, len(perm), 128):
            b = perm[i:i + 128]
            x = Xtr[b]
            if torch.rand(1).item() < 0.5:          # symétrie gauche/droite de l'arène
                x = x.flip(-1)
                x[:, 4] = Xtr[b][:, 4]
                y = (Ytr[b] // COLS) * COLS + (COLS - 1 - Ytr[b] % COLS)
            else:
                y = Ytr[b]
            logit = net(x, Ctr[b]).flatten(1).masked_fill(~Mtr[b].flatten(1), -1e9)
            loss = (torch.nn.functional.cross_entropy(logit, y, reduction="none") * Wtr[b]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        if epoch % 10 == 9:
            print(f"époque {epoch + 1} : perte {tot / len(Xtr):.3f}", flush=True)

    # --- test sur les vidéos jamais vues ---
    net.eval()
    Xte, Cte, Yte, Mte, _ = tensors(test)
    with torch.no_grad():
        logit = net(Xte, Cte).flatten(1).masked_fill(~Mte.flatten(1), -1e9)
    pred = logit.argmax(1)
    pc, pr = (pred % COLS).numpy(), (pred // COLS).numpy()
    tc, tr = (Yte % COLS).numpy(), (Yte // COLS).numpy()
    dist = np.hypot(pc - tc, pr - tr)
    lane = (pc < 9) == (tc < 9)
    print(f"\nTEST (vidéos jamais vues, toutes cartes) : écart médian {np.median(dist):.1f} cases, "
          f"à 2 cases ou moins {np.mean(dist <= 2):.0%}, même couloir {np.mean(lane):.0%}")
    # comparaison avec les règles actuelles, sur nos cartes
    rules = {(r["video"], round(r["t"], 1), r["card"]): r for r in
             (json.loads(l) for l in open(ROOT / "runs/videos/placement_compare.jsonl", encoding="utf-8"))}
    ours = [(i, rules.get((m["vid"], round(m["t"], 1), m["card"]))) for i, m in enumerate(test)]
    ours = [(i, r) for i, r in ours if r and r.get("ai")]
    if ours:
        idx = [i for i, _ in ours]
        rd = [np.hypot(r["ai"][0] - r["pro"][0], r["ai"][1] - r["pro"][1]) for _, r in ours]
        rl = [(r["ai"][0] < 9) == (r["pro"][0] < 9) for _, r in ours]
        print(f"NOS CARTES ({len(ours)} coups) — modèle : écart {np.median(dist[idx]):.1f} c, même couloir {np.mean(lane[idx]):.0%}"
              f"  |  règles actuelles : écart {np.median(rd):.1f} c, même couloir {np.mean(rl):.0%}")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), MODEL_DIR / "model.pt")
    (MODEL_DIR / "meta.json").write_text(json.dumps({"cards": cards, "moves": len(moves),
                                                     "test_median_gap": float(np.median(dist)),
                                                     "test_same_lane": float(np.mean(lane))}, indent=1), encoding="utf-8")
    print("modèle enregistré :", MODEL_DIR)


if __name__ == "__main__":
    main()
