"""Bandit factorisé de clashai/strategy.py : compatibilité du fichier, choix, apprentissage simulé.
STATS est redirigé vers tmp_path : les vrais fichiers ne sont jamais touchés."""
import json
import random
import shutil
from pathlib import Path

import pytest

from clashai import strategy as ST

REAL = Path(__file__).resolve().parents[1] / "learning/strategy_stats.json"


@pytest.fixture(autouse=True)
def tmp_stats(tmp_path, monkeypatch):
    path = tmp_path / "runs/strategy_stats.json"
    monkeypatch.setattr(ST, "STATS", path)
    return path


def entry(params, wins=0, losses=0):
    return {"params": params, "wins": wins, "losses": losses}


def totals(stats):
    return sum(v["wins"] for v in stats.values()), sum(v["losses"] for v in stats.values())


def test_space_and_default_agree():
    assert set(ST.SPACE) == set(ST.DEFAULT)
    for k in ST.SPACE:
        assert ST.DEFAULT[k] in ST.SPACE[k], k
    assert ST.SPACE["spell_value"] == [0.6, 0.8, 1.0] and ST.DEFAULT["spell_value"] == 0.8
    assert 0.06 in ST.SPACE["defend_line"]


def test_load_real_file_backward_compatible(tmp_stats):
    raw_text = REAL.read_text(encoding="utf-8")
    raw = json.loads(raw_text)
    seed = tmp_stats.parents[1] / "learning/strategy_stats.json"
    seed.parent.mkdir(parents=True)
    shutil.copy(REAL, seed)

    stats = ST.load()
    assert tmp_stats.read_text(encoding="utf-8") == raw_text      # graine copiée telle quelle dans runs/
    assert totals(stats) == totals(raw)                            # aucune partie perdue ni inventée
    assert 0 < len(stats) <= len(raw)
    for k, v in stats.items():
        assert set(v) == {"params", "wins", "losses"}
        assert set(ST.DEFAULT) <= set(v["params"])                 # complétées par DEFAULT
        assert k == ST._key(v["params"])
        assert v["wins"] + v["losses"] > 0                         # plus de variantes 0/0 injectées
    assert REAL.read_text(encoding="utf-8") == raw_text

    # le fichier existe déjà : pas recopié, relu et toujours compatible
    p = ST.choose(stats, rng=random.Random(0))
    ST.record(stats, p, "win")
    again = ST.load()
    assert totals(again) == (totals(raw)[0] + 1, totals(raw)[1])


def test_load_without_any_file_is_empty(tmp_stats):
    assert ST.load() == {}
    assert set(ST.choose({}, rng=random.Random(0))) == set(ST.DEFAULT)


def test_missing_keys_completed_and_merged():
    full = ST.complete({"giant_elixir": 8})
    stats = {
        ST._key({"giant_elixir": 8}): entry({"giant_elixir": 8}, 2, 1),   # ancienne variante incomplète
        ST._key(full): entry(full, 1, 1),                                  # la même, complète
        ST._key(ST.DEFAULT): entry(dict(ST.DEFAULT)),                      # 0/0 : oubliée
    }
    merged = ST.normalize(stats)
    assert list(merged) == [ST._key(full)]
    assert merged[ST._key(full)]["wins"] == 3 and merged[ST._key(full)]["losses"] == 2

    # statistiques par paramètre : identiques avec ou sans fusion préalable
    table, table_merged = ST.param_table(stats), ST.param_table(merged)
    assert table == table_merged
    row = {r[0]: r for r in table["giant_elixir"]}[8]
    assert row[1:3] == (3, 2)
    # fonction ajoutée après ces parties : elles comptent pour « désactivée » (LEGACY), pas pour DEFAULT
    assert full["placement_model"] is False and full["stat_defense"] is False
    assert {r[0]: r for r in table["spell_value"]}[ST.LEGACY["spell_value"]][1:3] == (3, 2)
    assert {r[0]: r for r in table["spell_value"]}[ST.DEFAULT["spell_value"]][1:3] == (0, 0)
    # une valeur héritée n'est jamais proposée
    assert ST.choose(stats, rng=random.Random(3))["spell_value"] in ST.SPACE["spell_value"]

    lb = ST.leaderboard(stats)
    assert len(lb) == 1 and lb[0][1:] == (3, 2, full)


def test_choose_complete_in_space_and_registered(tmp_stats):
    stats = json.loads(REAL.read_text(encoding="utf-8"))
    rng = random.Random(3)
    for _ in range(30):
        p = ST.choose(stats, rng=rng)
        assert set(p) == set(ST.DEFAULT)
        for k, v in p.items():
            assert v in ST.SPACE[k], (k, v)
        assert stats[ST._key(p)]["params"] == p
    before = dict(stats[ST._key(p)])
    ST.record(stats, p, "loss")
    assert stats[ST._key(p)]["losses"] == before["losses"] + 1
    assert json.loads(tmp_stats.read_text(encoding="utf-8"))[ST._key(p)]["losses"] == before["losses"] + 1
    assert set(ST.choose(stats)) == set(ST.DEFAULT)                # appel par défaut (autoplay)


def test_choose_is_deterministic_with_rng():
    stats = json.loads(REAL.read_text(encoding="utf-8"))
    a = ST.choose(json.loads(json.dumps(stats)), rng=random.Random(42))
    b = ST.choose(json.loads(json.dumps(stats)), rng=random.Random(42))
    assert a == b
    m1, m2 = ST.mutate(ST.DEFAULT, random.Random(7)), ST.mutate(ST.DEFAULT, random.Random(7))
    assert m1 == m2 and m1 != ST.DEFAULT and set(m1) == set(ST.DEFAULT)


def _exploit_stats():
    """Valeurs par défaut très gagnantes ; pour chaque paramètre une valeur « peu essayée » désignée."""
    stats = {ST._key(ST.DEFAULT): entry(dict(ST.DEFAULT), 1000, 0)}
    rare = {}
    for k, vals in ST.SPACE.items():
        others = [v for v in vals if v != ST.DEFAULT[k]]
        rare[k] = others[-1]
        for v in others:
            p = dict(ST.DEFAULT, **{k: v})
            stats[ST._key(p)] = entry(p, 0, 10 if v == rare[k] else 100)
    return stats, rare


def test_explore_forces_least_tested_value():
    stats, rare = _exploit_stats()
    assert ST.choose(stats, explore=0.0, rng=random.Random(1)) == ST.DEFAULT
    for seed in range(10):
        p = ST.choose(stats, explore=1.0, rng=random.Random(seed))
        diff = {k: v for k, v in p.items() if v != ST.DEFAULT[k]}
        assert len(diff) == 1
        (k, v), = diff.items()
        assert v == rare[k]


def test_record_counts_win_loss_ignores_draw(tmp_stats):
    stats = {}
    p = ST.choose(stats, rng=random.Random(0))
    for result in ("win", "draw", "loss", "loss", "timeout"):
        ST.record(stats, p, result)
    assert stats[ST._key(p)]["wins"] == 1 and stats[ST._key(p)]["losses"] == 2
    assert json.loads(tmp_stats.read_text(encoding="utf-8")) == stats
    # paramètres incomplets : comptés avec la variante complète équivalente
    drop = next(k for k in ST.DEFAULT if k not in ST.LEGACY and p[k] == ST.DEFAULT[k])
    ST.record(stats, {k: v for k, v in p.items() if k != drop}, "win")
    assert stats[ST._key(p)]["wins"] == 2


def test_leaderboard_and_param_table_shapes():
    stats = json.loads(REAL.read_text(encoding="utf-8"))
    lb = ST.leaderboard(stats)
    assert lb and all(len(r) == 4 for r in lb)
    assert [r[0] for r in lb] == sorted((r[0] for r in lb), reverse=True)
    for est, w, l, p in lb:
        assert 0 < est < 1 and isinstance(w, int) and isinstance(l, int) and w + l > 0
        assert set(ST.DEFAULT) <= set(p)
    assert (sum(r[1] for r in lb), sum(r[2] for r in lb)) == totals(stats)

    table = ST.param_table(stats)
    assert set(table) == set(ST.SPACE)
    for k, rows in table.items():
        assert all(len(r) == 4 and 0 < r[3] < 1 for r in rows)
        assert [r[3] for r in rows] == sorted((r[3] for r in rows), reverse=True)
        assert set(ST.SPACE[k]) <= {r[0] for r in rows}
        assert (sum(r[1] for r in rows), sum(r[2] for r in rows)) == totals(stats)   # chaque partie une fois
    assert 10 in {r[0] for r in table["giant_elixir"]}               # ancienne valeur rapportée…
    assert all(ST.choose(stats, rng=random.Random(s))["giant_elixir"] != 10 for s in range(20))   # …mais plus jouée


# --- apprentissage simulé : une valeur gagne 75 %, toutes les autres 40 %

def _old_choose(stats, rng, explore=0.45):
    """Ancienne méthode : Thompson sur les combinaisons entières, Beta(v+1, d+1), mutations."""
    draws = {k: rng.betavariate(v["wins"] + 1, v["losses"] + 1) for k, v in stats.items()}
    best = max(draws, key=draws.get)
    if rng.random() < explore:
        new = ST.mutate(stats[best]["params"], rng)
        stats.setdefault(ST._key(new), entry(new))
        return new
    return stats[best]["params"]


def _simulate(chooser, seed, param, good, games=150, last=50):
    rng, env = random.Random(seed), random.Random(seed + 1000)
    stats = {ST._key(ST.DEFAULT): entry(dict(ST.DEFAULT))}   # départ de l'ancien load() sans fichier
    picks = []
    for _ in range(games):
        p = chooser(stats, rng)
        won = env.random() < (0.75 if p[param] == good else 0.40)
        ST.record(stats, p, "win" if won else "loss")
        picks.append(p[param] == good)
    return sum(picks[-last:]) / last


@pytest.mark.parametrize("param,good", [("giant_spot", "mid"), ("cycle_at", 8.0), ("counter_push", False)])
def test_factorized_learns_faster_than_whole_combinations(param, good):
    assert good != ST.DEFAULT[param]
    seeds = range(10)                            # 3 graines : trop de bruit (moyenne 0.79 ou 0.84 selon les tirages)
    new = [_simulate(lambda s, r: ST.choose(s, rng=r), seed, param, good) for seed in seeds]
    old = [_simulate(_old_choose, seed, param, good) for seed in seeds]
    assert min(new) >= 0.6, new                  # la bonne valeur est jouée la plupart du temps
    assert sum(new) / len(new) >= 0.8, new
    assert sum(new) / len(new) > sum(old) / len(old) + 0.15, (new, old)


def test_archetype_of_opponent_decks():
    assert ST.archetype(["mortar", "knight", "archers", "zap"]) == "siege"
    assert ST.archetype(["minions", "baby-dragon", "knight", "zap"]) == "air"
    assert ST.archetype(["golem", "knight", "archers", "zap"]) == "tank"
    assert ST.archetype(["hog-rider", "bandit", "knight", "zap"]) == "bridge"
    assert ST.archetype(["knight", "archers", "zap", "fireball"]) == "mixed"


def test_choose_for_context_follows_what_won_against_that_type():
    stats = {}
    p = ST.choose(stats, rng=random.Random(0))
    ctx_stats = {}
    good = dict(p, giant_elixir=9)
    bad = dict(p, giant_elixir=7)
    for _ in range(15):
        ctx_stats.setdefault("bridge", {})
        e1 = ctx_stats["bridge"].setdefault(ST._key(ST.complete(good)), {"params": ST.complete(good), "wins": 0, "losses": 0})
        e1["wins"] += 1
        e2 = ctx_stats["bridge"].setdefault(ST._key(ST.complete(bad)), {"params": ST.complete(bad), "wins": 0, "losses": 0})
        e2["losses"] += 1
    picks = [ST.choose_for("bridge", stats, ctx_stats, rng=random.Random(s))["giant_elixir"] for s in range(20)]
    assert picks.count(9) > picks.count(7)
    for k, v in ST.PINNED.items():
        assert ST.choose_for("bridge", stats, ctx_stats)[k] == v


def test_load_ctx_seeds_from_learning(tmp_path, monkeypatch):
    seed = Path(__file__).resolve().parents[1] / "learning/strategy_ctx.json"
    if not seed.exists():
        pytest.skip("pas d'état par type d'adversaire")
    runs = tmp_path / "runs"
    monkeypatch.setattr(ST, "STATS", runs / "strategy_stats.json")
    monkeypatch.setattr(ST, "CTX_STATS", runs / "strategy_ctx.json")
    fake_root = tmp_path
    (fake_root / "learning").mkdir()
    (fake_root / "learning/strategy_ctx.json").write_text(seed.read_text(encoding="utf-8"), encoding="utf-8")
    assert ST.load_ctx() and (runs / "strategy_ctx.json").exists()     # avant : {} sur une machine neuve
