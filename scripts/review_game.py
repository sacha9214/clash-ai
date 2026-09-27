"""Bilan rapide d'un match joué par l'IA : ce qui a marché, ce qui cloche (détection, décisions, lenteur).

  .venv\\Scripts\\python scripts/review_game.py            (dernier match)
  .venv\\Scripts\\python scripts/review_game.py 20260927-083411
"""
from __future__ import annotations

import collections
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    games = sorted(p for p in (ROOT / "runs/games").iterdir() if p.is_dir() and (p / "decisions.jsonl").exists())
    g = next((p for p in games if p.name == sys.argv[1]), None) if len(sys.argv) > 1 else games[-1]
    L = [json.loads(l) for l in open(g / "decisions.jsonl", encoding="utf-8")]
    J = [json.loads(l) for l in open(ROOT / "runs/games/journal.jsonl", encoding="utf-8")]
    res = next((j for j in J if j.get("game") == g.name), {})
    opp = json.loads((g / "opponent.json").read_text(encoding="utf-8")) if (g / "opponent.json").exists() else {}
    deck = set(res.get("enemy_deck") or [])
    from clashai.opponent import SPAWNER_OF, UNIT2CARD
    print(f"=== {g.name} : {res.get('result', '?')} — {len(L)} coups, {res.get('refused', '?')} refusés ===")
    print("deck adverse vu :", ", ".join(sorted(deck)) or "?")
    played = [p["card"] for p in opp.get("played", [])]
    if played:
        t0 = opp["played"][0]["t"]
        fast = sum(1 for a, b in zip(opp["played"], opp["played"][1:]) if b["t"] - a["t"] < 1.0)
        print(f"cartes adverses comptées : {len(played)} (dont {fast} à moins d'1 s de la précédente : suspect)")
    if L:
        print(f"élixir médian au moment de jouer : {st.median(r['elixir'] for r in L):.1f}")
        print("cartes jouées :", dict(collections.Counter(r["card"] for r in L)))
    # décisions prises contre une unité absente du deck adverse (fausse détection)
    ghosts = collections.Counter()
    for r in L:
        if r["reason"].startswith("défense : "):
            u = r["reason"][len("défense : "):].split(" ")[0]
            card = UNIT2CARD.get(u, (u,))[0]
            spawned = any(u in SPAWNER_OF.get(c, ()) for c in deck)   # Squelettes de sa Pierre tombale, etc.
            if deck and card not in deck and not spawned and u not in ("goblin-barrel",):
                ghosts[u] += 1
    if ghosts:
        print("⚠ défenses contre des unités ABSENTES de son deck (fausses détections) :", dict(ghosts))
    spells1 = sum(1 for r in L if "groupe de 1" in r["reason"])
    if spells1:
        print(f"⚠ sorts sur une seule cible : {spells1}")
    perf = g / "perf.jsonl"
    if perf.exists():
        P = [json.loads(l) for l in open(perf, encoding="utf-8")]
        if P:
            print(f"boucle : médiane {st.median(p['loop_ms'] for p in P)} ms, détection {st.median(p['det_ms'] for p in P)} ms, "
                  f"pire {max(p['loop_ms'] for p in P)} ms")
    reasons = collections.Counter(r["reason"].split(" ->")[0].split(" (")[0].split(" [")[0] for r in L)
    print("raisons principales :", reasons.most_common(6))


if __name__ == "__main__":
    main()
