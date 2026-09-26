"""Où poser une carte : modèle appris sur les coups des joueurs des vidéos (imitation, phase 2).

Le plateau est vu du côté de celui qui joue (son camp en bas) sur la grille du jeu (18 x 32 cases) :
unités ennemies / alliées par case, tanks et volants ennemis, position. La carte module le réseau (FiLM).
Sortie : une carte de probabilité sur les cases ; on ne garde que les cases autorisées (notre moitié pour une
troupe, partout pour un sort). Entraîné par scripts/train_placement.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

COLS, ROWS = 18, 32
OWN_FIRST_ROW = 17
MODEL_DIR = Path(__file__).resolve().parents[1] / "models/placement"
TANKS = {"giant", "golem", "pekka", "mega-knight", "royal-giant", "electro-giant", "goblin-giant", "lava-hound",
         "giant-skeleton", "elixir-golem-big", "balloon", "hog-rider", "hog", "ram-rider", "battle-ram"}
AIR = {"minion", "bat", "mega-minion", "baby-dragon", "inferno-dragon", "balloon", "lava-hound", "lava-pup",
       "electro-dragon", "skeleton-dragon", "phoenix-big", "flying-machine"}
SPELLS = {"arrows", "fireball", "zap", "poison", "lightning", "rocket", "freeze", "rage", "clone", "tornado",
          "earthquake", "graveyard", "the-log", "giant-snowball", "barbarian-barrel", "royal-delivery", "goblin-barrel",
          "miner", "goblin-drill", "mirror"}
N_CH = 7


def features(units: list[tuple[str, bool, float, float]]) -> np.ndarray:
    """units : (nom, ennemi, colonne, rangée) en cases, vus du côté du joueur (son camp en bas)."""
    x = np.zeros((N_CH, ROWS, COLS), np.float32)
    for name, enemy, tx, ty in units:
        c, r = int(np.clip(tx, 0, COLS - 1)), int(np.clip(ty, 0, ROWS - 1))
        x[0 if enemy else 1, r, c] += 1
        if enemy and name in TANKS:
            x[2, r, c] += 1
        if enemy and name in AIR:
            x[3, r, c] += 1
    x[4] = np.linspace(0, 1, COLS)[None, :]                     # position (le réseau sait où il est)
    x[5] = np.linspace(0, 1, ROWS)[:, None]
    x[6, OWN_FIRST_ROW:, :] = 1                                 # notre moitié
    return x


def legal_mask(card: str) -> np.ndarray:
    m = np.zeros((ROWS, COLS), bool)
    if card in SPELLS:
        m[:] = True
    else:
        m[OWN_FIRST_ROW:, :] = True
    return m


def build_net(n_cards: int):
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(n_cards, 64)
            self.c1 = nn.Conv2d(N_CH, 48, 3, padding=1)
            self.c2 = nn.Conv2d(48, 48, 3, padding=2, dilation=2)
            self.film = nn.Linear(64, 96)
            self.c3 = nn.Conv2d(48, 48, 3, padding=4, dilation=4)
            self.c4 = nn.Conv2d(48, 1, 1)

        def forward(self, x, card):
            h = torch.relu(self.c1(x))
            h = torch.relu(self.c2(h))
            g, b = self.film(self.emb(card)).chunk(2, dim=1)       # la carte module les caractéristiques
            h = h * (1 + g[:, :, None, None]) + b[:, :, None, None]
            h = torch.relu(self.c3(h))
            return self.c4(h)[:, 0]                                 # (B, ROWS, COLS) logits

    return Net()


class PlacementModel:
    """Chargé une fois ; predict() renvoie la meilleure case autorisée (colonne, rangée) et la carte de proba."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        import torch
        meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
        self.cards = {c: i for i, c in enumerate(meta["cards"])}
        self.net = build_net(len(self.cards))
        self.net.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
        self.net.eval()
        self.torch = torch

    def knows(self, card: str) -> bool:
        return card in self.cards

    def predict(self, card: str, units, forbid=None) -> tuple[tuple[int, int], np.ndarray]:
        torch = self.torch
        with torch.no_grad():
            x = torch.from_numpy(features(units))[None]
            logit = self.net(x, torch.tensor([self.cards[card]]))[0].numpy()
        mask = legal_mask(card) if forbid is None else legal_mask(card) & ~forbid
        logit = np.where(mask, logit, -1e9)
        p = np.exp(logit - logit.max())
        p /= p.sum()
        r, c = np.unravel_index(int(p.argmax()), p.shape)
        return (int(c), int(r)), p


def load() -> PlacementModel | None:
    if not (MODEL_DIR / "model.pt").exists():
        return None
    try:
        return PlacementModel()
    except Exception:
        return None
