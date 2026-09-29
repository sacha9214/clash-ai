"""Cerveau : scénarios tirés des vrais matchs (learning/games_logs.tgz), sans téléphone."""
import math

import pytest

from clashai import card_info
from clashai.brain import OWN_FIRST_ROW, PHONE, Brain, Decision, Seen

TH = PHONE.th     # une case en hauteur (fraction d'écran)
ALL = [True, True, True, True]


def at(col, row):
    return PHONE.center(col, row)


def enemy(name, col, row, vy_tiles=0.0):
    x, y = at(col, row)
    return Seen(name, True, x, y, 0.0, vy_tiles * TH)


def ours(name, col, row, vy_tiles=0.0):
    x, y = at(col, row)
    return Seen(name, False, x, y, 0.0, vy_tiles * TH)


def brain(**p):
    return Brain({"placement_model": False, **p})


# ---- statistiques : toutes les cartes à la même échelle ----
def test_rarity_normalised_stats():
    # valeurs « niveau 1 » des fichiers = premier niveau de la rareté (rare = niveau 3 : x1.21)
    assert card_info.combat("mini-pekka")["hp"] == pytest.approx(642 / 1.21, rel=0.01)
    assert card_info.combat("knight")["hp"] == pytest.approx(690, rel=0.01)          # commune : inchangée
    assert card_info.spell_damage("fireball") == pytest.approx(325 / 1.21, rel=0.01)
    assert card_info.spell_damage("arrows") == pytest.approx(48 * 3, rel=0.01)       # 3 vagues


# ---- sorts ----
def test_no_fireball_on_lone_tank_even_if_a_support_unit_exists_elsewhere():
    # 25/09 : 57 Boules de feu sur un tank seul, parce qu'un tireur existait AILLEURS sur le terrain
    b = brain(fireball_min=1)
    seen = [enemy("giant", 3, 20, vy_tiles=0.8), enemy("musketeer", 9, 8)]
    d = b.decide(seen, ["fireball", "arrows", "knight", "archers"], ALL, 7.0, 100.0)
    if d and d.card == "fireball":
        tx, ty = PHONE.to_tile(d.x, d.y)
        assert math.hypot(tx - 9.5, ty - 8.5) < 1.5, "la Boule de feu doit viser le Mousquetaire, pas le Géant"


def test_lone_giant_gets_a_defender_not_a_spell():
    b = brain(fireball_min=1)
    d = b.decide([enemy("giant", 3, 20, vy_tiles=0.8)], ["fireball", "arrows", "knight", "mini-pekka"], ALL, 7.0, 100.0)
    assert d is not None and d.card not in ("fireball", "arrows")


def test_fireball_on_lone_musketeer_is_a_good_trade():
    b = brain(fireball_min=1)
    d = b.decide([enemy("musketeer", 9, 8)], ["fireball", "giant", "knight", "archers"], [True, False, False, False],
                 4.5, 100.0)
    assert d is not None and d.card == "fireball"


def test_fireball_group_musketeer_knight():
    b = brain(fireball_min=2)
    seen = [enemy("musketeer", 4, 8), enemy("knight", 5, 9)]
    d = b.decide(seen, ["fireball", "giant", "knight", "archers"], [True, False, False, False], 4.5, 100.0)
    assert d is not None and d.card == "fireball"


def test_arrows_on_minions_but_not_on_skeletons_far_away(old_deck):
    b = brain(arrows_min=3)
    minions = [enemy("minion", 9 + i % 2, 8 + i // 2) for i in range(3)]
    d = b.decide(minions, ["arrows", "giant", "knight", "valkyrie"], [True, False, False, False], 3.5, 100.0)
    assert d is not None and d.card == "arrows"
    skel = [enemy("skeleton", 9 + i % 2, 8 + i // 2) for i in range(3)]
    d = brain(arrows_min=3).decide(skel, ["arrows", "giant", "knight", "valkyrie"], [True, False, False, False], 3.5, 100.0)
    assert d is None


def test_arrows_on_goblins_hitting_our_side(old_deck):
    b = brain(arrows_min=3)
    gob = [enemy("goblin", 3 + i % 2, 22 + i // 2) for i in range(3)]
    d = b.decide(gob, ["arrows", "giant", "cannon", "fireball"], [True, False, False, False], 3.5, 100.0)
    assert d is not None and d.card == "arrows"


def test_spells_never_touch_the_enemy_king():
    b = brain(arrows_min=3)
    near_king = [enemy("minion", 8 + i, 5) for i in range(3)]     # juste sous le Roi (rangées 0.5-4.5)
    d = b.decide(near_king, ["arrows", "fireball", "giant", "knight"], [True, True, False, False], 8.0, 100.0)
    assert d is None or d.card not in ("arrows", "fireball")


# ---- placements précis gardés malgré le modèle des pros ----
def test_cannon_rule_position_survives_placement_model():
    b = Brain({"placement_model": True})
    if b.placer is None:
        pytest.skip("modèle de placement absent")
    d = b.decide([enemy("giant", 3, 18, vy_tiles=0.8)], ["cannon", "fireball", "arrows", "giant"],
                 [True, True, True, False], 5.0, 100.0)
    assert d is not None and d.card == "cannon"
    assert d.tile[0] in (8, 9) and d.tile[1] == OWN_FIRST_ROW + 6
    assert d.precise


def test_non_precise_decision_goes_through_the_model():
    b = Brain({"placement_model": True})
    if b.placer is None:
        pytest.skip("modèle de placement absent")
    d = b.decide([], ["knight", "giant", "arrows", "fireball"], [True, False, True, True], 9.5, 100.0)
    assert d is not None and d.card == "knight" and "apprise des pros" in d.reason


# ---- mémoire : seulement les cartes vraiment posées ----
def test_decide_has_no_side_effect_until_played():
    b = brain()
    d = b.decide([], ["knight", "giant", "arrows", "fireball"], [True, False, True, True], 9.5, 100.0)
    assert d is not None
    for _ in range(10):                                   # la boucle redécide pendant le délai entre deux cartes
        b.decide([], ["knight", "giant", "arrows", "fireball"], [True, False, True, True], 9.5, 100.1)
    assert b.virtual == [] and b.own_recent == []
    b.played(d, 100.2)
    assert len(b.virtual) == 1 and b.own_played["knight"] == 100.2


def test_virtual_troop_dies_under_fire():
    b = brain()
    d = Decision("knight", 0, *at(3, 20), "test")
    b.played(d, 0.0)
    pekkas = [enemy("mini-pekka", 3, 19) for _ in range(3)]
    alive_at = None
    for i in range(1, 140):
        b._fix_sides(pekkas, i * 0.1)
        v = b._virtual_units(pekkas, i * 0.1)
        if not v:
            alive_at = i * 0.1
            break
    assert alive_at is not None and alive_at < 6, "un Chevalier contre 3 Mini P.E.K.K.A ne survit pas 14 s"


def test_virtual_troop_without_enemies_lives_on():
    b = brain()
    b.played(Decision("knight", 0, *at(3, 28), "test"), 0.0)
    assert b._virtual_units([], 5.0)


# ---- camp des unités ----
def test_unplayed_own_giant_beyond_river_walking_down_is_an_enemy():
    b = brain()
    s = b._fix_sides([ours("giant", 14, 12, vy_tiles=1.0)], 100.0)
    assert len(s) == 1 and s[0].enemy


def test_unplayed_own_unit_standing_beyond_river_is_ignored():
    b = brain()
    assert b._fix_sides([ours("giant", 14, 12)], 100.0) == []


def test_our_giant_that_we_played_stays_ours():
    b = brain()
    b.played(Decision("giant", 0, *at(14, 28), "attaque"), 80.0)
    s = b._fix_sides([ours("giant", 14, 12, vy_tiles=-0.8)], 100.0)
    assert len(s) == 1 and not s[0].enemy


def test_no_phantom_support_behind_misread_enemy_giant():
    # 25/09 20:05 : Géant ennemi lu « à nous » au pont -> « soutien : mini-pekka derrière le Géant »
    b = brain()
    d = b.decide([ours("giant", 14, 13, vy_tiles=1.0)], ["mini-pekka", "knight", "arrows", "fireball"], ALL, 6.8, 100.0)
    assert d is None or not d.reason.startswith("soutien")


# ---- élixir ----
def test_cycle_threshold_is_reachable():
    b = brain(cycle_at=10.0)                              # la barre se lit au plus ~9.5
    d = b.decide([], ["knight", "arrows", "fireball", "cannon"], [True, True, True, True], 9.5, 100.0)
    assert d is not None


def test_full_elixir_plays_a_kept_counter_instead_of_leaking():
    b = brain()
    b.opp_hand = ["giant", "arrows"]                      # sa grosse menace peut revenir : on garde Canon/Mini P.E.K.K.A
    d = b.decide([], ["cannon", "mini-pekka", "arrows", "fireball"], ALL, 9.5, 100.0)
    assert d is not None and d.card in ("cannon", "mini-pekka")
    d = b.decide([], ["cannon", "mini-pekka", "arrows", "fireball"], ALL, 6.0, 100.0)
    assert d is None                                      # sans élixir plein, on les garde


# ---- vitesse ----
class _FakeUnit:
    def __init__(self, name, box, vel, enemy=True, track_id=1):
        self.name, self.box, self.vel, self.enemy, self.track_id = name, box, vel, enemy, track_id


def test_to_seen_uses_detector_velocity():
    u = _FakeUnit("knight", (100, 400, 140, 460), (0.0, 64.0))
    (s,) = Brain.to_seen([u], 578, 1280, {1: [(0, 0)] * 10}, fps=1.0)   # traînée et fps absurdes : ignorés
    assert s.vy == pytest.approx(64.0 / 1280)


def test_duel_mini_pekka_beats_knight():
    win, _, _ = Brain._duel("mini-pekka", [enemy("knight", 3, 20)], near_tower=False)
    assert win


def test_lone_giant_is_answered_by_mini_pekka_not_the_cheapest_card():
    # un Géant ne frappe que les bâtiments : toutes les cartes « gagnaient » le duel et les Archères (17 s) passaient
    b = brain()
    d = b.decide([enemy("giant", 3, 19, vy_tiles=0.8)], ["knight", "mini-pekka", "archers", "valkyrie"], ALL, 6.0, 100.0)
    assert d is not None and d.card == "mini-pekka"
    assert not Brain._duel("knight", [enemy("giant", 3, 19)], near_tower=False)[0]


def test_evolved_goblin_barrel_gets_the_valkyrie_and_ui_bars_are_not_units():
    barrel = _FakeUnit("goblin-barrel-evolution", (100, 700, 140, 740), (0.0, 0.0))
    bar = _FakeUnit("skeleton-king-bar", (300, 300, 340, 310), (0.0, 0.0))
    seen = Brain.to_seen([barrel, bar], 578, 1280, None, fps=0)
    assert [s.name for s in seen] == ["goblin-barrel"]
    d = brain().decide(seen, ["valkyrie", "knight", "arrows", "giant"], ALL, 6.0, 100.0)
    assert d is not None and d.card == "valkyrie"


# ---- avance d'élixir et bilan des échanges ----
def test_ahead_launches_the_giant_earlier():
    b = brain(edge_push=3, giant_elixir=9)
    b.opp_elixir = 1.5                                   # il vient de tout dépenser
    d = b.decide([], ["giant", "arrows", "fireball", "cannon"], ALL, 8.0, 100.0)
    assert d is not None and d.card == "giant" and "avance" in d.reason
    b0 = brain(edge_push=0, giant_elixir=9)
    b0.opp_elixir = 1.5
    assert b0.decide([], ["giant", "arrows", "fireball", "cannon"], ALL, 8.0, 100.0) is None


def test_behind_holds_the_giant():
    b = brain(edge_push=3, giant_elixir=7)
    b.opp_elixir = 10.0
    seen = [enemy("golem", 9, 3)]                        # il prépare un Golem au fond (8 élixir sur le terrain)
    d = b.decide(seen, ["giant", "arrows", "fireball", "cannon"], ALL, 8.0, 100.0)
    assert d is None or d.card != "giant"
    d = brain(edge_push=0, giant_elixir=7).decide(seen, ["giant", "arrows", "fireball", "cannon"], ALL, 8.0, 100.0)
    assert d is not None and d.card == "giant"


def test_our_giant_on_the_board_counts_so_we_still_support_it():
    b = brain(edge_push=3, support_min_elixir=4)
    b.opp_elixir = 9.0                                   # 4 en main mais un Géant (5) qui avance : pas « en retard »
    b.played(Decision("giant", 0, *at(3, 28), "attaque : Géant"), 90.0)
    d = b.decide([ours("giant", 3, 20, vy_tiles=-0.8)], ["mini-pekka", "arrows", "fireball", "cannon"], ALL, 7.0, 100.0)
    assert d is not None and d.reason.startswith("soutien")


def test_support_keeps_elixir_to_defend_the_other_lane():
    b = brain(edge_push=3, support_min_elixir=4)
    b.opp_elixir = 9.0                                   # il a de quoi contre-attaquer de l'autre côté
    b.played(Decision("giant", 0, *at(3, 28), "attaque : Géant"), 90.0)
    d = b.decide([ours("giant", 3, 20, vy_tiles=-0.8)], ["mini-pekka", "arrows", "fireball", "cannon"], ALL, 4.0, 100.0)
    assert d is None or not d.reason.startswith("soutien")   # 4 - 4 = 0 : on garde 3 pour défendre


def test_trade_mini_pekka_on_giant_and_push_credited_once():
    b = brain()
    d = b.decide([enemy("giant", 3, 19, vy_tiles=0.8)], ["knight", "mini-pekka", "archers", "valkyrie"], ALL, 6.0, 100.0)
    assert d.card == "mini-pekka" and d.trade == pytest.approx(5.0, abs=0.01)   # il tue le Géant sans une égratignure
    b.played(d, 100.0)
    d2 = Decision("knight", 1, *at(3, 22), "défense : giant -> knight", trade=2.0, push=5.0)
    b.played(d2, 103.0)                                  # 2e carte contre le même Géant : l'attaque est déjà comptée
    assert b.trade_balance == pytest.approx(5.0 - 3.0, abs=0.01)


# ---- fin de match selon les couronnes ----
from clashai.towers import MatchState  # noqa: E402


def _match(elapsed, ours=0, theirs=0, now=1000.0):
    m = MatchState(start=now - elapsed)
    m._enemy_seen = {lane: (now - 10 if lane < ours else now) for lane in (0, 1)}     # tour absente 10 s = tombée
    m.our_hp = {lane: (0.0 if lane < theirs else 0.8) for lane in (0, 1)}
    return m


HAND = ["giant", "arrows", "fireball", "cannon"]


def test_leading_late_stops_attacking():
    b = brain(endgame=True, edge_push=0, giant_elixir=7)
    b.match = _match(160, ours=1)
    d = b.decide([], HAND, ALL, 8.0, 1000.0)
    assert d is None or d.card != "giant"
    b.match = _match(100, ours=1)                          # pas encore la fin : on attaque normalement
    assert b.decide([], HAND, ALL, 8.0, 1000.0).card == "giant"


def test_trailing_late_goes_all_in():
    b = brain(endgame=True, edge_push=0, giant_elixir=9)
    b.match = _match(165, theirs=1)
    d = b.decide([], HAND, ALL, 5.5, 1000.0)
    assert d is not None and d.card == "giant" and "attaque" in d.reason
    b0 = brain(endgame=False, edge_push=0, giant_elixir=9)
    b0.match = _match(165, theirs=1)
    assert b0.decide([], HAND, ALL, 5.5, 1000.0) is None


def test_overtime_is_sudden_death():
    b = brain(endgame=True, edge_push=0, giant_elixir=9)
    b.match = _match(200)
    assert b._endgame(1000.0) == "mort subite"
    d = b.decide([], HAND, ALL, 7.2, 1000.0)
    assert d is not None and d.card == "giant"


def test_hidden_enemy_tower_is_not_a_crown_too_soon():
    m = _match(160)
    m._enemy_seen[0] = 1000.0 - 4                          # masquée 4 s par des unités : pas encore une couronne
    assert m.crowns(1000.0) == (0, 0)


# ---- achever une tour ----
def _low_tower_match(frac, ok=True, now=1000.0):
    m = _match(60, now=now)
    m.enemy_hp, m.enemy_hp_ok = {0: frac, 1: 1.0}, ok
    m.enemy_alive = {0: True, 1: True}
    return m


def test_fireball_finishes_an_almost_dead_tower_without_touching_the_king():
    b = brain()
    b.match = _low_tower_match(0.04)
    d = b.decide([], ["fireball", "knight", "giant", "cannon"], [True, False, False, False], 4.5, 1000.0)
    assert d is not None and d.card == "fireball" and "achève" in d.reason
    from clashai.brain import _hits_enemy_king
    assert not _hits_enemy_king("fireball", d.x, d.y)


def test_arrows_finish_off_centre_and_not_on_a_healthy_tower(old_deck):
    b = brain()
    b.match = _low_tower_match(0.02)
    d = b.decide([], ["arrows", "knight", "giant", "cannon"], [True, False, False, False], 3.5, 1000.0)
    from clashai.brain import _hits_enemy_king
    assert d is not None and d.card == "arrows" and not _hits_enemy_king("arrows", d.x, d.y)
    b.match = _low_tower_match(0.30)
    assert b.decide([], ["arrows", "knight", "giant", "cannon"], [True, False, False, False], 3.5, 1000.0) is None


def test_no_finish_when_hp_reading_is_unproven():
    b = brain()
    b.match = _low_tower_match(0.04, ok=False)
    assert b.decide([], ["fireball", "knight", "giant", "cannon"], [True, False, False, False], 4.5, 1000.0) is None


# ---- contre-attaque avec les survivants ----
def test_counter_push_behind_a_surviving_defender(old_deck):
    b = brain(counter_support=True, edge_push=0)
    b.played(Decision("knight", 0, *at(3, 22), "défense : goblin -> knight", trade=1.0, push=2.0), 95.0)
    survivor = ours("knight", 3, 20, vy_tiles=-1.0)          # il repart vers le pont
    d = b.decide([survivor], ["archers", "fireball", "arrows", "cannon"], ALL, 6.0, 100.0)
    assert d is not None and d.card == "archers" and d.reason.startswith("contre-attaque")
    assert d.y > survivor.y                                    # derrière lui (vers notre Roi)


def test_no_counter_push_without_recent_defense_or_with_an_enemy_still_here():
    hand = ["archers", "fireball", "arrows", "cannon"]
    b = brain(counter_support=True, edge_push=0)
    assert b.decide([ours("knight", 3, 20, vy_tiles=-1.0)], hand, ALL, 5.0, 100.0) is None   # pas de défense récente
    b.played(Decision("knight", 0, *at(3, 22), "défense : goblin -> knight"), 95.0)
    d = b.decide([ours("knight", 3, 20, vy_tiles=-1.0), enemy("skeleton", 14, 24)], hand, ALL, 5.0, 100.0)
    assert d is None or not d.reason.startswith("contre-attaque")
    b0 = brain(counter_support=False, edge_push=0)
    b0.played(Decision("knight", 0, *at(3, 22), "défense : goblin -> knight"), 95.0)
    assert b0.decide([ours("knight", 3, 20, vy_tiles=-1.0)], hand, ALL, 5.0, 100.0) is None


def test_unknown_stats_threat_is_still_defended():
    # Boss Bandit : PV/dégâts absents de la base -> ne doit pas passer pour « inoffensif »
    d = brain().decide([enemy("boss-bandit", 3, 20, vy_tiles=1.0)], ["knight", "mini-pekka", "arrows", "fireball"],
                       ALL, 6.0, 100.0)
    assert d is not None and d.reason.startswith("défense")


def test_air_attack_without_anti_air_gets_an_offset_decoy():
    b = brain()
    t = enemy("mega-minion", 3, 22)
    d = b.decide([t], ["valkyrie", "giant", "cannon", "fireball"], ALL, 8.0, 100.0)
    assert d is not None and d.card == "valkyrie" and "appât" in d.reason
    assert abs(d.x - t.x) > 0.08                         # pas collée à l'unité volante : décalée vers le centre


def test_no_ground_only_card_against_flyers():
    b = brain()
    d = b.decide([enemy("minion", 3, 24), enemy("minion", 4, 24), enemy("minion", 3, 25)],
                 ["mini-pekka", "giant", "cannon", "fireball"], ALL, 8.0, 100.0)
    assert d is None or d.card != "mini-pekka"


import pytest


@pytest.fixture
def old_deck(monkeypatch):
    """Archères et Flèches (ancien deck, sur le banc depuis le 27/09) : leurs règles restent testées."""
    from clashai.cards import BENCH, DECK
    for c in ("archers", "arrows"):
        monkeypatch.setitem(DECK, c, BENCH[c])


def test_our_minions_read_as_enemy_are_ours_if_he_has_no_minions():
    b = brain()
    b.opp_deck = ["giant", "pekka", "zap", "fireball"]
    b.played(Decision("minions", 0, *at(3, 24), "test"), 90.0)
    d = b.decide([enemy("minion", 3, 12, vy_tiles=-1.0)], ["musketeer", "knight", "cannon", "fireball"], ALL, 8.0, 100.0)
    assert d is None or not d.reason.startswith("défense : minion")


def test_no_cannon_against_minions_even_with_a_ground_unit_nearby():
    b = brain()
    seen = [enemy("minion", 3, 24), enemy("minion", 4, 24), enemy("goblin", 4, 23)]
    d = b.decide(seen, ["cannon", "knight", "giant", "fireball"], ALL, 8.0, 100.0)
    assert d is None or d.card not in ("cannon",) or "goblin" in d.reason


def test_our_fresh_minions_misread_as_enemy_are_ours():
    b = brain()
    b.played(Decision("minions", 0, *at(4, 20), "test"), 96.0)
    d = b.decide([enemy("minion", 4, 19), enemy("minion", 5, 19)], ["knight", "valkyrie", "cannon", "fireball"],
                 ALL, 8.0, 100.0)
    assert d is None or not d.reason.startswith("défense : minion")


def test_no_defense_against_his_buildings():
    b = brain()
    d = b.decide([enemy("goblin-cage", 4, 14)], ["minions", "knight", "cannon", "musketeer"], ALL, 6.0, 100.0)
    assert d is None or not d.reason.startswith("défense : goblin-cage")


def test_duplicate_enemy_copy_of_our_giant_is_ignored():
    b = brain()
    d = b.decide([ours("giant", 3, 20), enemy("giant", 3, 20)], ["cannon", "minions", "knight", "fireball"], ALL, 8.0, 100.0)
    assert d is None or not d.reason.startswith("défense : giant")


def test_pekka_read_as_his_mini_pekka_when_he_has_no_pekka():
    b = brain()
    b.opp_deck = ["mini-pekka", "giant", "minions", "zap"]
    d = b.decide([enemy("pekka", 3, 22)], ["knight", "cannon", "minions", "fireball"], ALL, 8.0, 100.0)
    assert d is None or "pekka x" not in d.reason.replace("mini-pekka", "")


def test_waits_a_moment_for_valkyrie_against_a_crowd():
    b = brain()
    seen = [enemy("barbarian", 3, 20 + i % 2) for i in range(5)]
    d = b.decide(seen, ["musketeer", "mini-pekka", "valkyrie", "knight"], ALL, 3.0, 100.0)
    assert d is None


def test_our_musketeer_hitting_his_tower_is_not_defended():
    b = brain()
    b.opp_deck = ["dark-prince", "hog-rider", "valkyrie", "knight"]
    b.played(Decision("musketeer", 0, *at(3, 24), "test"), 80.0)
    d = b.decide([enemy("musketeer", 3, 13, vy_tiles=0.2)], ["knight", "cannon", "minions", "fireball"], ALL, 8.0, 100.0)
    assert d is None or not d.reason.startswith("défense : musketeer")


def test_tower_handles_a_few_skeletons():
    b = brain()
    d = b.decide([enemy("skeleton", 3, 24), enemy("skeleton", 4, 24)], ["valkyrie", "minions", "knight", "cannon"],
                 ALL, 6.0, 100.0)
    assert d is None or not d.reason.startswith("défense")


def test_our_giant_walking_up_our_half_read_as_enemy_is_ours():
    b = brain()
    b.played(Decision("giant", 0, *at(3, 19), "test"), 87.0)
    b.virtual.clear()
    d = b.decide([enemy("giant", 3, 18, vy_tiles=-0.6)], ["cannon", "minions", "knight", "fireball"], ALL, 8.0, 100.0)
    assert d is None or not d.reason.startswith("défense : giant")


# ---- revue du 27/09 soir ----
def test_three_minions_at_our_tower_are_defended():
    # 90 PV chacune : la tour ne les tue pas en 1 coup (3 x 46 dégâts/s) -> on défend
    seen = [enemy("minion", 3 + i % 2, 22 + i // 2) for i in range(3)]
    d = brain().decide(seen, ["musketeer", "knight", "giant", "cannon"], ALL, 6.0, 100.0)
    assert d is not None and d.reason.startswith(("défense", "fireball"))


def test_two_skeletons_still_left_to_the_tower():
    seen = [enemy("skeleton", 3, 22), enemy("skeleton", 4, 22)]
    assert brain().decide(seen, ["musketeer", "knight", "giant", "cannon"], ALL, 6.0, 100.0) is None


def test_first_real_goblin_barrel_answered_with_six_cards_known():
    b = brain()
    b.opp_deck = ["knight", "giant", "musketeer", "minions", "valkyrie", "cannon"]   # 6 connues, pas le Tonneau
    barrel = Seen("goblin-barrel", True, *at(3, 24))
    d = b.decide([barrel], ["valkyrie", "knight", "giant", "cannon"], ALL, 6.0, 100.0)
    assert d is not None and d.card == "valkyrie"


def test_his_first_mirror_knight_in_the_other_lane_stays_enemy():
    b = brain()
    b.opp_deck = ["giant", "musketeer", "minions", "valkyrie"]
    b.played(Decision("knight", 0, *at(3, 24), "défense : x"), 90.0)      # notre Chevalier, couloir gauche
    s = b._fix_sides([enemy("knight", 14, 20, vy_tiles=1.0)], 100.0)       # le sien, couloir droit
    assert len(s) == 1 and s[0].enemy


def test_our_minions_from_the_centre_read_as_enemy_in_a_lane_are_ours():
    b = brain()
    b.played(Decision("minions", 0, *at(9, 28), "test"), 95.0)
    d = b.decide([enemy("minion", 4, 22), enemy("minion", 4, 23)], ["knight", "cannon", "valkyrie", "fireball"],
                 ALL, 8.0, 100.0)
    assert d is None or not d.reason.startswith("défense : minion")


def test_evolved_musketeer_snipes_a_lane_push_from_behind_the_tower():
    b = brain()
    b.musk_plays = 2                                     # 3e pose : évoluée
    seen = [enemy("giant", 3, 20), enemy("wizard", 3, 17)]
    d = b.decide(seen, ["musketeer", "knight", "cannon", "fireball"], ALL, 8.0, 100.0)
    assert d is not None and d.card == "musketeer" and "évoluée" in d.reason
    from clashai.brain import OWN_TOWER_Y
    assert d.y > OWN_TOWER_Y                             # derrière la tour


def test_normal_musketeer_counts_toward_evolution():
    b = brain()
    for i in range(2):
        b.played(Decision("musketeer", 0, *at(3, 26), "test"), 50.0 + i)
    assert b.evo_musketeer()
    b.played(Decision("musketeer", 0, *at(3, 26), "test"), 60.0)
    assert not b.evo_musketeer()


def test_giant_only_with_elixir_to_support_it():
    b = brain(giant_elixir=7, edge_push=0)
    d = b.decide([], ["giant", "knight", "musketeer", "cannon"], ALL, 7.0, 100.0)
    assert d is None or d.card != "giant"                # 7 - 5 = 2 : pas de quoi le soutenir
    d = b.decide([], ["giant", "knight", "musketeer", "cannon"], ALL, 8.2, 100.0)
    assert d is not None and d.card == "giant"


def test_counter_push_keeps_a_defense_reserve():
    b = brain(counter_support=True, edge_push=0)
    b.played(Decision("knight", 0, *at(3, 22), "défense : goblin -> knight", trade=1.0, push=2.0), 95.0)
    d = b.decide([ours("knight", 3, 20, vy_tiles=-1.0)], ["musketeer", "fireball", "minions", "cannon"], ALL, 5.0, 100.0)
    assert d is None or not d.reason.startswith("contre-attaque")
