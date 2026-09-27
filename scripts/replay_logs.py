"""Rejoue les décisions enregistrées des vrais matchs avec le cerveau ACTUEL (sans téléphone) et compare.

Chaque ligne de runs/games/*/decisions.jsonl garde les unités vues, la main et l'élixir au moment du coup :
on redemande au cerveau ce qu'il ferait (mêmes paramètres que ce match) et on compte ce qui change.
L'élixir, le deck et la main de l'adversaire sont reconstitués depuis opponent.json (cartes vues, élixir après chacune).
Limites : pas de vitesses ni d'état des tours dans les journaux.
  --set edge_push=0 : force un réglage (comparer deux variantes du cerveau sur les mêmes situations).

  python scripts/replay_logs.py                       # runs/games (ou learning/games_logs.tgz extrait)
  python scripts/replay_logs.py --games D:/logs/runs/games --show 20
"""
import argparse
import collections
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clashai.brain import Brain  # noqa: E402
from clashai.cards import TANK_UNITS  # noqa: E402

W, H = 578, 1280


class _Unit:
    def __init__(self, name, enemy, center):
        cx, cy = center
        self.name, self.enemy, self.track_id, self.vel = name, enemy, -1, (0.0, 0.0)
        self.box = (cx - 15, cy - 20, cx + 15, cy + 20)


REGEN = 1 / 2.8


def _opponent_at(opp: list[dict], t: float, t0: float) -> tuple[float, list[str], list[str]]:
    """(élixir, deck, main probable) de l'adversaire à l'instant t, d'après les cartes vues avant t."""
    before = [e for e in opp if e["t"] <= t]
    rate = REGEN * (2 if t - t0 > 120 else 1)
    elixir = min(10.0, before[-1]["elixir_after"] + (t - before[-1]["t"]) * rate) if before else \
        min(10.0, 5 + (t - t0) * rate)
    played = [e["card"] for e in before]
    deck = list(dict.fromkeys(played))[:8]
    return elixir, deck, [c for c in deck if c not in played[-4:]]


def _value(v: str):
    try:
        return json.loads(v)
    except ValueError:
        return v


def _kind(reason: str) -> str:
    return re.sub(r"\s*[\(\[].*|\s*->.*|\d+", "", reason).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="runs/games")
    ap.add_argument("--journal", default="learning/journal.jsonl")
    ap.add_argument("--no-model", action="store_true", help="règles seules (sans le modèle de placement)")
    ap.add_argument("--show", type=int, default=10, help="exemples de décisions changées à afficher")
    ap.add_argument("--set", action="append", default=[], metavar="CLE=VALEUR", help="force un paramètre du cerveau")
    a = ap.parse_args()
    params = {}
    if os.path.exists(a.journal):
        for line in open(a.journal, encoding="utf-8"):
            r = json.loads(line)
            params[r["game"]] = r.get("params") or {}
    same = changed = 0
    tank_fireballs = collections.Counter()
    edge_moves = collections.Counter()
    trades = []
    transitions = collections.Counter()
    examples = []
    for path in sorted(glob.glob(os.path.join(a.games, "*", "decisions.jsonl"))):
        game = os.path.basename(os.path.dirname(path))
        p = dict(params.get(game, {}))
        if a.no_model:
            p["placement_model"] = False
        p.update({k: _value(v) for k, v in (kv.split("=", 1) for kv in a.set)})
        brain = Brain(p)
        opp_path = os.path.join(os.path.dirname(path), "opponent.json")
        opp = json.load(open(opp_path, encoding="utf-8"))["played"] if os.path.exists(opp_path) else []
        rows = [json.loads(line) for line in open(path, encoding="utf-8")]
        t0 = min([r["t"] for r in rows[:1]] + [e["t"] for e in opp[:1]]) - 3 if rows else 0
        for r in rows:
            brain.opp_elixir, brain.opp_deck, brain.opp_hand = _opponent_at(opp, r["t"], t0)
            units = [_Unit(n, e, c) for n, e, c in r["units"]]
            seen = Brain.to_seen(units, W, H, None, 0)
            hand = list(r["hand"])
            d = brain.decide(seen, hand, [c is not None for c in hand], r["elixir"], r["t"])
            new = d.card if d else None
            for who, card, reason in (("avant", r["card"], r["reason"]), ("après", new, d.reason if d else "")):
                if card == "fireball" and reason.startswith("fireball sur un groupe de 1"):
                    near = [u for u in units if u.enemy and "tower" not in u.name]
                    x, y = (r["x"], r["y"]) if who == "avant" else (d.x, d.y)
                    target = min(near, key=lambda u: abs(u.box[0] / W + 15 / W - x) + abs(u.box[3] / H - y), default=None)
                    tank_fireballs[who] += bool(target and target.name in TANK_UNITS)
            if new == r["card"]:
                same += 1
            else:
                changed += 1
                transitions[(_kind(r["reason"]), _kind(d.reason) if d else "(attend)")] += 1
                if len(examples) < a.show:
                    examples.append(f"  {game} {r['reason'][:60]:60} -> {d.reason[:60] if d else '(attend)'}")
            if d:
                if "avance d'élixir" in d.reason:
                    edge_moves["Géant plus tôt (avance)"] += 1
                brain.played(d, r["t"])
            elif r["card"] == "giant" and brain._edge(r["elixir"])[0] < 0:
                edge_moves["Géant retenu (retard)"] += 1
        trades.append(brain.trade_balance)
    total = same + changed
    print(f"{total} décisions rejouées : {same} identiques ({same / max(total, 1):.0%}), {changed} changées")
    print(f"Boules de feu sur un tank seul : avant {tank_fireballs['avant']}, après {tank_fireballs['après']}")
    if edge_moves:
        print("Avance d'élixir : " + ", ".join(f"{k} {n}" for k, n in edge_moves.items()))
    if trades:
        print(f"Bilan des échanges estimé (coups du cerveau actuel) : médiane {sorted(trades)[len(trades) // 2]:+.1f} "
              f"élixir par match")
    print("Changements les plus fréquents :")
    for (old, new), n in transitions.most_common(12):
        print(f"  {n:4d}  {old}  ->  {new}")
    if examples:
        print("Exemples :")
        print("\n".join(examples))


if __name__ == "__main__":
    main()
