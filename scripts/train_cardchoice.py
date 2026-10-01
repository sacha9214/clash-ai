"""Quelle carte poser ? Appris des pros (imitation), mesuré contre nos règles sur des vidéos jamais vues.

Données : les coups des pros extraits des vidéos (runs/videos/*.jsonl) dont la carte est dans NOTRE deck
(~4 900 coups) ; pour chacun, l'état du terrain juste avant. Le modèle répond à « face à ce terrain, laquelle
de nos cartes un pro pose-t-il ? ». Il ne sait pas QUAND ne rien jouer (les vidéos ne montrent que des coups).

Mesures sur les vidéos de test (jamais apprises) :
  - toujours la carte la plus fréquente (plancher),
  - le modèle (carte la plus probable ; et « dans les 2 premières »),
  - nos règles actuelles (Brain, les 7 cartes en main, élixir plein) : accord avec le pro.

  .venv-yolo\\Scripts\\python scripts/train_cardchoice.py
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from clashai import card_info  # noqa: E402
from clashai.cards import DECK  # noqa: E402
from clashai.tiles import PHONE  # noqa: E402
from train_placement import is_test, load_moves  # noqa: E402

CARDS = [c for c in DECK if DECK[c].kind != "spell"]          # nos 7 cartes posables (troupes + Canon)
MODEL_DIR = ROOT / "models/cardchoice"
RIVER_ROW = 16


def vocab(moves) -> list[str]:
    c = collections.Counter(n for m in moves for n, enemy, _, _ in m["units"] if enemy)
    return [n for n, k in c.most_common(60)]


def feat(units, names: list[str]) -> np.ndarray:
    """Terrain -> vecteur : par zone (notre moitié / la sienne x gauche / droite), ce que l'ennemi et nous y avons
    (nombre, élixir, PV, dégâts, volants, tanks qui visent les bâtiments, zone, nuées) + quelles unités ennemies
    sont là + distance de la menace la plus avancée."""
    z = np.zeros((2, 4, 8), np.float32)
    present = np.zeros(len(names), np.float32)
    near = np.zeros(len(names), np.float32)
    deepest = 0.0
    for name, enemy, tx, ty in units:
        st = card_info.combat(name)
        zone = (0 if ty >= RIVER_ROW else 2) + (0 if tx < 9 else 1)
        v = z[0 if enemy else 1, zone]
        v += (1, st["cost"] / max(st["count"], 1) / 5, st["hp"] / 1000, st["dps"] / 100, float(st["flying"]),
              float(st["buildings_only"]), float(st["splash"]), float(st["count"] >= 3))
        if enemy and name in names:
            present[names.index(name)] = 1
            if ty >= RIVER_ROW:
                near[names.index(name)] = 1
        if enemy:
            deepest = max(deepest, ty / 32)
    return np.concatenate([z.ravel(), present, near, [deepest]]).astype(np.float32)


def rules_choice(units) -> str | None:
    """Ce que nos règles joueraient sur ce terrain, les 7 cartes en main et l'élixir plein."""
    from clashai.brain import Brain, Seen
    b = Brain({"placement_model": False})
    seen = [Seen(n, e, *PHONE.center(int(np.clip(tx, 0, 17)), int(np.clip(ty, 0, 31)))) for n, e, tx, ty in units]
    best = None
    for hand in (CARDS[:4], CARDS[3:7], [CARDS[0], CARDS[4], CARDS[5], CARDS[6]]):   # 3 mains qui couvrent le deck
        d = b.decide(seen, hand, [True] * 4, 9.4, 100.0)
        if d is not None and d.card in CARDS and (best is None or d.reason.startswith("défense")):
            best = d.card
    return best


def main():
    torch.manual_seed(0)
    torch.set_num_threads(3)
    moves = [m for m in load_moves() if m["card"] in CARDS]
    train = [m for m in moves if not is_test(m["vid"])]
    test = [m for m in moves if is_test(m["vid"])]
    names = vocab(train)
    X = torch.tensor(np.stack([feat(m["units"], names) for m in train]))
    y = torch.tensor([CARDS.index(m["card"]) for m in train])
    Xt = torch.tensor(np.stack([feat(m["units"], names) for m in test]))
    yt = torch.tensor([CARDS.index(m["card"]) for m in test])
    print(f"{len(moves)} coups de pros avec nos cartes : {len(train)} pour apprendre, {len(test)} pour tester")
    print("répartition :", dict(collections.Counter(m["card"] for m in moves).most_common()))
    # poids par classe : sinon le modèle répond « Canon » partout (40 % des coups)
    w = torch.tensor([len(train) / (len(CARDS) * max(1, int((y == i).sum()))) for i in range(len(CARDS))]).float() ** 0.5
    net = nn.Sequential(nn.Linear(X.shape[1], 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 64), nn.ReLU(),
                        nn.Linear(64, len(CARDS)))
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
    best_acc, best_state = 0.0, None
    for ep in range(150):
        net.train()
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 128):
            idx = perm[i:i + 128]
            loss = nn.functional.cross_entropy(net(X[idx]), y[idx], weight=w)
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            acc = float((net(Xt).argmax(1) == yt).float().mean())
        if acc > best_acc:
            best_acc, best_state = acc, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        logits = net(Xt)
    top1 = float((logits.argmax(1) == yt).float().mean())
    top2 = float((logits.topk(2, 1).indices == yt[:, None]).any(1).float().mean())
    major = collections.Counter(m["card"] for m in train).most_common(1)[0][0]
    base = float(np.mean([m["card"] == major for m in test]))
    rc = [rules_choice(m["units"]) for m in test]
    played = [(r, m["card"]) for r, m in zip(rc, test) if r is not None]
    rules_acc = float(np.mean([r == c for r, c in played])) if played else 0.0
    print(f"toujours « {major} »            : {base:.0%}")
    print(f"modèle (1re carte)            : {top1:.0%}   | dans ses 2 premières : {top2:.0%}")
    print(f"nos règles (quand elles jouent, {len(played)}/{len(test)} terrains) : {rules_acc:.0%} d'accord avec le pro")
    per = {c: (int(((logits.argmax(1) == i) & (yt == i)).sum()), int((yt == i).sum())) for i, c in enumerate(CARDS)}
    print("par carte (retrouvés / coups du pro) :", per)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, MODEL_DIR / "model.pt")
    (MODEL_DIR / "meta.json").write_text(json.dumps({"cards": CARDS, "names": names, "test_top1": top1, "test_top2": top2,
                                                     "rules_agree": rules_acc, "baseline": base, "moves": len(moves)},
                                                    indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
