"""Apprentissage des stratégies : à chaque combat on essaie une variante, on retient
lesquelles gagnent (bandit de Thompson : on joue surtout les meilleures, mais on
continue d'explorer les autres et on en invente de nouvelles en mutant les gagnantes).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

STATS = Path(__file__).resolve().parents[1] / "runs/strategy_stats.json"

# Paramètres réglables du cerveau et leurs plages
SPACE = {
    "giant_elixir": [7, 8, 9, 10],          # élixir minimum pour lancer le Géant
    "giant_spot": ["back", "mid", "bridge"],   # Géant au fond (grosse poussée), au milieu, ou au pont (pression)
    "support_min_elixir": [3, 4, 5],        # soutien derrière le Géant dès que…
    "arrows_min": [2, 3, 4],                # taille de groupe minimale pour Flèches
    "fireball_min": [1, 2, 3],              # … pour Boule de feu (1 = accepte une grosse cible seule)
    "defend_line": [0.0, 0.05, 0.10],       # on défend quand l'ennemi est à cette distance au-delà de la rivière
    "counter_push": [False, True],          # après une défense réussie, relancer dans le même couloir
    "cycle_at": [9.0, 9.5, 10.0],           # élixir plein : on fait tourner une carte à partir de…
    "punish_low_elixir": [False, True],     # attaquer dès que l'élixir estimé de l'ennemi est bas
    "fireball_spawners": [False, True],     # Boule de feu sur les bâtiments qui produisent des unités
    "ignore_small": [False, True],          # laisser les tours gérer 1-2 petites unités
}
DEFAULT = {"giant_elixir": 9, "giant_spot": "back", "support_min_elixir": 4, "arrows_min": 3,
           "fireball_min": 2, "defend_line": 0.06, "counter_push": False, "cycle_at": 9.5,
           "punish_low_elixir": False, "fireball_spawners": False,
           "ignore_small": True}


def _key(p: dict) -> str:
    return json.dumps(p, sort_keys=True)


def load() -> dict:
    if STATS.exists():
        return json.loads(STATS.read_text())
    return {_key(DEFAULT): {"params": DEFAULT, "wins": 0, "losses": 0}}


def save(stats: dict) -> None:
    STATS.parent.mkdir(parents=True, exist_ok=True)
    STATS.write_text(json.dumps(stats, indent=1))


def mutate(p: dict) -> dict:
    q = dict(DEFAULT, **p)
    for k in random.sample(list(SPACE), k=random.choice([1, 2])):
        q[k] = random.choice([v for v in SPACE[k] if v != q[k]])
    return q


def choose(stats: dict, explore: float = 0.45) -> dict:
    """Thompson : tirer une proba de victoire Beta(v+1, d+1) par variante, prendre la meilleure.
    Parfois (explore) : nouvelle variante, mutation de la meilleure actuelle."""
    draws = {k: random.betavariate(v["wins"] + 1, v["losses"] + 1) for k, v in stats.items()}
    best = max(draws, key=draws.get)
    if random.random() < explore:
        new = mutate(stats[best]["params"])
        stats.setdefault(_key(new), {"params": new, "wins": 0, "losses": 0})
        return new
    return stats[best]["params"]


def record(stats: dict, params: dict, result: str) -> None:
    s = stats.setdefault(_key(params), {"params": params, "wins": 0, "losses": 0})
    if result == "win":
        s["wins"] += 1
    elif result == "loss":
        s["losses"] += 1
    save(stats)


def leaderboard(stats: dict) -> list[tuple[float, int, int, dict]]:
    rows = [((v["wins"] + 1) / (v["wins"] + v["losses"] + 2), v["wins"], v["losses"], v["params"])
            for v in stats.values()]
    return sorted(rows, key=lambda r: -r[0])
