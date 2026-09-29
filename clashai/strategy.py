"""Apprentissage des stratégies : à chaque combat on joue une combinaison de paramètres et on retient
ce qui gagne. Bandit de Thompson factorisé : chaque paramètre est appris séparément (toutes les parties
jouées avec la valeur v comptent pour v), donc chaque combat renseigne tous les paramètres à la fois,
au lieu d'une seule combinaison parmi des milliers qui ne serait presque jamais rejouée.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

STATS = Path(__file__).resolve().parents[1] / "runs/strategy_stats.json"
PRIOR_N = 2.0   # a priori = taux de victoire global, pesant 2 parties : une valeur jamais jouée est essayée sans tout écraser

# Paramètres réglables du cerveau et leurs plages (la valeur de DEFAULT doit y figurer)
SPACE = {
    "giant_elixir": [7, 8, 9],              # élixir minimum pour lancer le Géant (la barre se lit au plus ~9.5 : pas 10)
    "giant_spot": ["king", "back", "corner", "mid", "bridge"],   # derrière le Roi, au fond côté, dans le coin, au milieu, au pont
    "support_min_elixir": [3, 4, 5],        # soutien derrière le Géant dès que…
    "arrows_min": [2, 3, 4],                # taille de groupe minimale pour Flèches
    "fireball_min": [1, 2, 3],              # … pour Boule de feu (1 = accepte une grosse cible seule)
    "defend_line": [0.0, 0.05, 0.06, 0.10],  # on défend quand l'ennemi est à cette distance au-delà de la rivière
    "counter_push": [False, True],          # après une défense réussie, relancer dans le même couloir
    "cycle_at": [7.0, 8.0, 9.0, 9.5],       # rien à faire : on fait tourner une carte au fond à partir de… (pas 10 : illisible)
    "punish_low_elixir": [False, True],     # attaquer dès que l'élixir estimé de l'ennemi est bas
    "fireball_spawners": [False, True],     # Boule de feu sur les bâtiments qui produisent des unités
    "ignore_small": [False, True],          # laisser les tours gérer 1-2 petites unités
    # règles tirées des vidéos de stratégie (2026-09-25), à valider en match :
    "punish_opposite": [False, True],       # l'ennemi pose une carte lourde -> Mini P.E.K.K.A/Chevalier au pont, autre couloir
    "giant_when_counter_out": [False, True],  # Géant plus tôt quand ses contres connus ne sont plus dans sa main
    "fireball_patient": [False, True],      # Boule de feu sur cible seule : seulement si elle touche aussi une tour
    "placement_model": [True, False],       # « où poser » appris des pros (vidéos) plutôt que par les règles
    "stat_defense": [True, False],          # défenseur choisi par combat simulé (statistiques) plutôt que listes fixes
    "spell_value": [0.6, 0.8, 1.0],         # sort lancé seulement s'il détruit au moins spell_value x son coût en élixir
    "edge_push": [0, 3, 4],                 # avance d'élixir (main + terrain) pour attaquer tôt / retard : pas d'attaque ; 0 = ignoré
    "endgame": [False, True],               # 30 dernières s : en tête -> défendre, mené -> tout attaquer ; prolongation -> presser
    "finish_towers": [True, False],         # sort sur une tour ennemie presque morte (lecture des PV validée en match)
    "counter_support": [True, False],       # soutien derrière nos défenseurs survivants qui repartent vers le pont
}
DEFAULT = {"giant_elixir": 9, "giant_spot": "king", "support_min_elixir": 4, "arrows_min": 3,
           "fireball_min": 2, "defend_line": 0.06, "counter_push": True, "cycle_at": 9.5,
           "punish_low_elixir": False, "fireball_spawners": False,
           "ignore_small": True, "punish_opposite": False, "giant_when_counter_out": False,
           "fireball_patient": False, "placement_model": True, "stat_defense": True, "spell_value": 0.8,
           "edge_push": 3, "endgame": True, "finish_towers": True, "counter_support": True}


# Fonctions qui ont fait leurs preuves : plus tirées au hasard (27/09 : une défaite en 15 coups avec les trois
# coupées par l'exploration — Valkyrie sur des Squelettes, Mousquetaire sur un Gobelin)
PINNED = {"stat_defense": True, "ignore_small": True, "placement_model": True,
          "spell_value": 0.8}   # un sort détruit au moins 80 % de son coût (29/09 : Boule de feu pour 2,8 élixirs à 0.6)


def _key(p: dict) -> str:
    return json.dumps(p, sort_keys=True)


# Fonctions ajoutées après les premiers matchs (26-27/09) : une partie enregistrée SANS ce paramètre a été jouée
# sans la fonction -> elle compte pour « désactivé », pas pour la valeur par défaut d'aujourd'hui.
# spell_value 0.0 = pas de seuil de valeur (ancien comportement), valeur qui n'est plus proposée.
LEGACY = {"placement_model": False, "stat_defense": False, "spell_value": 0.0, "edge_push": 0, "endgame": False,
          "finish_towers": False, "counter_support": False}


def complete(p: dict) -> dict:
    """Paramètres complets : ceux qui n'existaient pas encore prennent leur valeur « ancienne » (LEGACY), sinon DEFAULT."""
    return {**DEFAULT, **LEGACY, **p}


def normalize(stats: dict) -> dict:
    """Même format, variantes complétées par DEFAULT, fusionnées si identiques ; celles sans partie sont oubliées."""
    out = {}
    for v in stats.values():
        p = complete(v["params"])
        s = out.setdefault(_key(p), {"params": p, "wins": 0, "losses": 0})
        s["wins"] += v.get("wins", 0)
        s["losses"] += v.get("losses", 0)
    return {k: v for k, v in out.items() if v["wins"] + v["losses"]}


def load() -> dict:
    seed = STATS.parents[1] / "learning/strategy_stats.json"   # état versionné sur GitHub
    if not STATS.exists() and seed.exists():
        STATS.parent.mkdir(parents=True, exist_ok=True)
        STATS.write_text(seed.read_text(encoding="utf-8"), encoding="utf-8")
    return normalize(json.loads(STATS.read_text(encoding="utf-8"))) if STATS.exists() else {}


def save(stats: dict) -> None:
    STATS.parent.mkdir(parents=True, exist_ok=True)
    STATS.write_text(json.dumps(stats, indent=1), encoding="utf-8")


def _counts(stats: dict) -> tuple[dict, int, int]:
    """{paramètre: {valeur: [victoires, défaites]}} sur toutes les variantes, + totaux."""
    c = {k: {} for k in SPACE}
    wins = losses = 0
    for v in stats.values():
        p, w, l = complete(v["params"]), v.get("wins", 0), v.get("losses", 0)
        wins, losses = wins + w, losses + l
        for k in SPACE:
            wl = c[k].setdefault(p[k], [0, 0])
            wl[0] += w
            wl[1] += l
    return c, wins, losses


def _prior(wins: int, losses: int) -> tuple[float, float]:
    """Beta(a, b) centrée sur le taux global : une valeur jamais jouée vaut « la moyenne, à peu près »."""
    p0 = (wins + 1) / (wins + losses + 2)
    return PRIOR_N * p0, PRIOR_N * (1 - p0)


def mutate(p: dict, rng: random.Random | None = None) -> dict:
    rng = rng or random
    q = complete(p)
    for k in rng.sample(list(SPACE), k=rng.choice([1, 2])):
        q[k] = rng.choice([v for v in SPACE[k] if v != q[k]])
    return q


def choose(stats: dict, explore: float = 0.1, rng: random.Random | None = None) -> dict:
    """Thompson par paramètre : pour chaque valeur possible, tirer une proba de victoire Beta(v + a, d + b)
    sur toutes les parties jouées avec cette valeur, garder la meilleure.
    Parfois (explore) : un paramètre au hasard prend sa valeur la moins essayée."""
    rng = rng or random
    counts, wins, losses = _counts(stats)
    a, b = _prior(wins, losses)
    p = dict(DEFAULT)
    for k, vals in SPACE.items():
        draws = [rng.betavariate(w + a, l + b) for w, l in (counts[k].get(v, (0, 0)) for v in vals)]
        p[k] = vals[draws.index(max(draws))]
    if rng.random() < explore:
        k = rng.choice([k for k in SPACE if k not in PINNED])
        tries = [sum(counts[k].get(v, (0, 0))) for v in SPACE[k]]
        p[k] = rng.choice([v for v, n in zip(SPACE[k], tries) if n == min(tries)])
    p.update(PINNED)
    stats.setdefault(_key(p), {"params": p, "wins": 0, "losses": 0})
    return p


def record(stats: dict, params: dict, result: str) -> None:
    p = complete(params)
    s = stats.setdefault(_key(p), {"params": p, "wins": 0, "losses": 0})
    if result == "win":
        s["wins"] += 1
    elif result == "loss":
        s["losses"] += 1
    save(stats)


def leaderboard(stats: dict) -> list[tuple[float, int, int, dict]]:
    """Combinaisons jouées, meilleure d'abord (même a priori que choose : peu de parties -> proche du taux global)."""
    stats = normalize(stats)
    a, b = _prior(sum(v["wins"] for v in stats.values()), sum(v["losses"] for v in stats.values()))
    rows = [((v["wins"] + a) / (v["wins"] + v["losses"] + a + b), v["wins"], v["losses"], v["params"])
            for v in stats.values()]
    return sorted(rows, key=lambda r: -r[0])


def param_table(stats: dict) -> dict[str, list[tuple[object, int, int, float]]]:
    """Par paramètre : [(valeur, victoires, défaites, proba de victoire a posteriori)], meilleure d'abord.
    Toutes les valeurs de SPACE (jamais jouée = a priori), plus les anciennes valeurs vues dans stats."""
    counts, wins, losses = _counts(stats)
    a, b = _prior(wins, losses)
    table = {}
    for k in SPACE:
        vals = list(SPACE[k]) + [v for v in counts[k] if v not in SPACE[k]]
        rows = [(v, *counts[k].get(v, (0, 0))) for v in vals]
        table[k] = sorted(((v, w, l, (w + a) / (w + l + a + b)) for v, w, l in rows), key=lambda r: -r[3])
    return table


# ---- apprendre PAR TYPE D'ADVERSAIRE : ce qui gagne contre un deck aérien n'est pas ce qui gagne contre un tank ----
CTX_STATS = STATS.with_name("strategy_ctx.json")
AIR_CARDS = {"minions", "minion-horde", "bats", "mega-minion", "baby-dragon", "skeleton-dragons", "electro-dragon",
             "inferno-dragon", "balloon", "lava-hound", "flying-machine", "phoenix"}
SWARM_CARDS = {"goblins", "spear-goblins", "goblin-gang", "skeletons", "skeleton-army", "barbarians", "rascals",
               "goblin-hut", "tombstone", "barbarian-hut", "furnace", "guards", "royal-recruits", "elite-barbarians"}
TANK_CARDS = {"golem", "giant", "pekka", "electro-giant", "goblin-giant", "royal-giant", "mega-knight", "lava-hound",
              "elixir-golem", "ice-golem"}
BRIDGE_CARDS = {"hog-rider", "battle-ram", "bandit", "royal-hogs", "ram-rider", "dark-prince", "prince", "mini-pekka",
                "lumberjack", "goblin-barrel", "wall-breakers"}
SIEGE_CARDS = {"mortar", "x-bow"}
CONTEXTS = ("siege", "air", "tank", "bridge", "swarm", "mixed")


def archetype(deck: list[str]) -> str:
    """Type du deck adverse (vu en cours de partie) : siège, aérien, tank, pression au pont, nuées, mixte."""
    d = set(deck)
    if d & SIEGE_CARDS:
        return "siege"
    if len(d & AIR_CARDS) >= 2:
        return "air"
    if d & (TANK_CARDS - {"ice-golem"}):
        return "tank"
    if len(d & BRIDGE_CARDS) >= 2:
        return "bridge"
    if len(d & SWARM_CARDS) >= 3:
        return "swarm"
    return "mixed"


def load_ctx() -> dict:
    return json.loads(CTX_STATS.read_text(encoding="utf-8")) if CTX_STATS.exists() else {}


def choose_for(ctx: str, stats: dict, ctx_stats: dict, rng: random.Random | None = None) -> dict:
    """Thompson par paramètre sur les parties CONTRE CE TYPE de deck ; a priori = taux global de chaque valeur
    (pesant PRIOR_N parties) : peu de parties de ce type -> on joue comme d'habitude, puis on s'en écarte."""
    rng = rng or random
    g, gw, gl = _counts(stats)
    c, _, _ = _counts(normalize(ctx_stats.get(ctx, {})))
    base = (gw + 1) / (gw + gl + 2)
    p = dict(DEFAULT)
    for k, vals in SPACE.items():
        best, pick = -1.0, vals[0]
        for v in vals:
            w0, l0 = g[k].get(v, (0, 0))
            rate = (w0 + PRIOR_N * base) / (w0 + l0 + PRIOR_N)
            w, l = c[k].get(v, (0, 0))
            x = rng.betavariate(w + PRIOR_N * rate, l + PRIOR_N * (1 - rate))
            if x > best:
                best, pick = x, v
        p[k] = pick
    p.update(PINNED)
    return p


def record_ctx(ctx_stats: dict, ctx: str, params: dict, result: str) -> None:
    s = ctx_stats.setdefault(ctx, {})
    p = complete(params)
    e = s.setdefault(_key(p), {"params": p, "wins": 0, "losses": 0})
    if result == "win":
        e["wins"] += 1
    elif result == "loss":
        e["losses"] += 1
    CTX_STATS.parent.mkdir(parents=True, exist_ok=True)
    tmp = CTX_STATS.with_suffix(".tmp")
    tmp.write_text(json.dumps(ctx_stats, indent=1), encoding="utf-8")
    os.replace(tmp, CTX_STATS)                        # jamais un fichier à moitié écrit si le PC s'arrête
