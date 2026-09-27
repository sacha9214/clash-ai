"""Modèle de l'adversaire (clashai/opponent.py) : scénarios image par image avec de fausses unités et des instants
explicites. Chaque test reproduit un défaut vu dans les vrais matchs (revue du 27/09)."""
import json
import os
from dataclasses import dataclass
from pathlib import Path

import pytest

from clashai.opponent import UNIT2CARD, Opponent

H = 1280
W = int(H / 2.2)          # l'image du flux : largeur ~ hauteur / 2,2
DT = 0.04                 # une image


@dataclass
class U:
    track_id: int
    name: str
    enemy: bool = True
    conf: float = 0.9
    box: tuple = (0, 0, 0, 0)
    coasted: bool = False
    vel: tuple = (0.0, 0.0)

    @property
    def center(self):
        return (self.box[0] + self.box[2]) // 2, (self.box[1] + self.box[3]) // 2


def unit(tid, name, x, y, **kw):
    """Unité centrée en (x, y), fractions de l'image (y = 0 : son camp, en haut)."""
    cx, cy = int(x * W), int(y * H)
    return U(tid, name, box=(cx - 20, cy - 25, cx + 20, cy + 25), **kw)


def opp():
    return Opponent(start=0.0, last_t=0.0)


def seen_twice(o, units, t):
    """Les mêmes unités sur deux images de suite : ce qu'il faut pour confirmer une pose."""
    return o.update(units, t, H) + o.update(units, t + DT, H)


def known_deck(o, cards):
    """Deck déjà connu : chaque carte vue deux fois."""
    o.played = [c for c in cards for _ in range(2)]


# ---- 1. plausible() : ne pas cacher de vraies unités au cerveau ----
def test_spawned_barbarians_of_his_battle_ram_are_plausible():
    o = opp()
    known_deck(o, ["battle-ram", "witch", "goblin-hut", "furnace", "knight", "musketeer", "valkyrie", "fireball"])
    assert len(o.deck) == 8
    # avant : barbares (sortis du Bélier), squelettes, gobelins à lance, esprits de feu cachés (pas dans le deck)
    for name in ("barbarian", "skeleton", "spear-goblin", "fire-spirit"):
        assert o.plausible(unit(50, name, 0.5, 0.6, conf=0.6)), name


def test_variants_spawned_and_unknown_classes_are_plausible_with_a_full_deck():
    o = opp()
    known_deck(o, ["knight", "musketeer", "valkyrie", "giant", "mini-pekka", "archers", "fireball", "arrows"])
    assert o.plausible(unit(1, "knight-evolution", 0.5, 0.6, conf=0.5))     # sa carte, en version évoluée
    assert o.plausible(unit(2, "golemite", 0.5, 0.6, conf=0.5))             # née d'une autre unité
    assert o.plausible(unit(3, "goblinstein", 0.5, 0.6, conf=0.5))          # classe hors table : invérifiable
    # carte hors deck : il suffit d'être sûr (plus de verrou « deck complet »)
    assert o.plausible(unit(4, "hog-rider", 0.5, 0.6, conf=0.8))
    assert not o.plausible(unit(5, "hog-rider", 0.5, 0.6, conf=0.6))
    assert o.plausible(unit(6, "knight", 0.5, 0.6, conf=0.3))               # carte du deck : toujours


def test_counted_play_stays_visible_when_its_confidence_drops():
    o = opp()
    hog = unit(9, "hog-rider", 0.5, 0.3, conf=0.7)
    assert seen_twice(o, [hog], 10.0) == ["hog-rider"]                      # 1re pose, pas encore « preuve » de deck
    hog.coasted, hog.conf = True, 0.63                                       # de mémoire : confiance x0,9
    assert o.plausible(hog)


# ---- 2. deck : preuves, pas « premier arrivé » ----
def test_early_ghosts_do_not_lock_out_a_strong_hog_rider():
    o = opp()
    t, tid = 1.0, 100
    for name in ["knight", "skeleton", "valkyrie", "elite-barbarian", "ice-golem", "barbarian", "bomber", "mini-pekka"]:
        seen_twice(o, [unit(tid, name, 0.2, 0.25, conf=0.65)], t)          # fausse détection vue une fois
        t, tid = t + 6, tid + 1
    assert len(o.deck) < 8
    hog = unit(999, "hog-rider", 0.7, 0.25, conf=0.95)
    assert seen_twice(o, [hog], t) == ["hog-rider"]
    assert "hog-rider" in o.deck and o.plausible(hog)


def test_better_supported_card_replaces_a_card_seen_once():
    o = opp()
    t, tid = 1.0, 1
    first8 = ["knight", "musketeer", "valkyrie", "giant", "mini-pekka", "archer", "bomber", "prince"]
    for name in first8:
        seen_twice(o, [unit(tid, name, 0.2, 0.25, conf=0.9)], t)
        t, tid = t + 7, tid + 1
    assert len(o.deck) == 8
    for _ in range(2):                                                       # une 9e carte vue deux fois (conf 0,8)
        assert seen_twice(o, [unit(tid, "hog-rider", 0.7, 0.25, conf=0.8)], t) == ["hog-rider"]
        t, tid = t + 30, tid + 1
    assert o.deck[0] == "hog-rider" and len(o.deck) == 8


def test_weak_unknown_card_is_noise():
    o = opp()
    assert seen_twice(o, [unit(1, "pekka", 0.5, 0.25, conf=0.5)], 10.0) == []
    assert o.played == [] and o.elixir == pytest.approx(5 + 10.04 / 2.8)


# ---- 3. une copie de mémoire ne confirme pas une pose ----
def test_coasted_copy_does_not_confirm_a_one_frame_ghost():
    o = opp()
    ghost = unit(7, "knight", 0.2, 0.25, conf=0.65)
    assert o.update([ghost], 10.0, H) == []
    ghost.coasted = True
    out = []
    for k in range(1, 21):                                                   # 0,8 s de mémoire du détecteur
        out += o.update([ghost], 10.0 + k * DT, H)
    assert out == [] and o.played == []


# ---- 4. sorts : les nôtres et les siens ----
def test_our_fireball_reidentified_3s_later_is_not_his():
    o = opp()
    o.note_our_spell(10.0, 0.5 * W, 0.55 * H)            # ancien appel (sans la carte) : règle de position
    assert o.update([unit(5, "fireball", 0.5, 0.55)], 13.1, H) == []
    assert o.played == []


def test_our_fireball_seen_on_its_way_up_is_not_his():
    o = opp()
    o.note_our_spell(10.0, 0.5 * W, 0.30 * H, card="fireball")      # visée chez lui, vue en partant de notre Roi
    out = []
    for k in range(9):
        out += o.update([unit(5, "fireball", 0.5, 0.62 - 0.04 * k)], 10.2 + 0.1 * k, H)
    out += o.update([unit(6, "fireball", 0.3, 0.55)], 12.1, H)              # re-suivi, loin de la cible
    assert out == []
    # sa Boule de feu à lui, 5 s après la nôtre, et ses Flèches pendant la nôtre : comptées
    assert o.update([unit(7, "fireball", 0.5, 0.6)], 15.0, H) == ["fireball"]
    o.note_our_spell(20.0, 0.5 * W, 0.3 * H, card="fireball")
    assert o.update([unit(8, "arrows", 0.5, 0.6)], 21.0, H) == ["arrows"]


def test_his_fireball_flying_from_his_half_into_ours_is_counted_once():
    o = opp()
    out = []
    for k in range(10):                                  # part de son Roi (y 0,15) et tombe sur notre tour (y 0,60)
        out += o.update([unit(5, "fireball", 0.45, 0.15 + 0.05 * k)], 10.0 + 0.1 * k, H)
    assert out == ["fireball"]                           # avant : jugé sur sa 1re image (chez lui), jamais compté
    assert o.update([unit(6, "fireball", 0.45, 0.6)], 11.2, H) == []       # re-suivi à l'impact : pas deux fois
    assert o.elixir == pytest.approx(5 + 11.2 / 2.8 - 4)


def test_spell_in_his_half_is_not_counted():
    o = opp()
    out = []
    for k in range(40):                                                      # notre sort chez lui, 1,6 s
        out += o.update([unit(5, "arrows", 0.5, 0.25)], 10.0 + k * DT, H)
    assert out == []


# ---- 5. camp lu à la confirmation, pas sur la 1re image ----
def test_side_flip_on_second_frame_is_still_counted():
    o = opp()
    out = []
    for k, enemy in enumerate([False, True, True, True]):                   # vote de camp : « à nous » puis badge rouge
        out += o.update([unit(11, "musketeer", 0.55, 0.28, conf=0.8, enemy=enemy)], 20.0 + k * DT, H)
    assert out == ["musketeer"]


def test_our_unit_born_in_his_half_is_not_his_play():
    o = opp()
    out = []
    for k in range(60):                                                      # notre pose chez lui (tour tombée)
        out += o.update([unit(12, "knight", 0.3, 0.3, enemy=False)], 20.0 + k * DT, H)
    assert out == []


# ---- 6. cycle : des poses ratées ne font pas rejeter les suivantes ----
def test_cycle_rule_accepts_a_card_again_after_missed_plays():
    o = opp()
    assert seen_twice(o, [unit(1, "knight", 0.3, 0.3)], 10.0) == ["knight"]
    assert seen_twice(o, [unit(2, "musketeer", 0.7, 0.3)], 20.0) == ["musketeer"]
    # 3 poses ratées entre-temps : son Chevalier revient, il faut le compter
    assert seen_twice(o, [unit(3, "knight", 0.7, 0.25)], 30.0) == ["knight"]
    assert o.played == ["knight", "musketeer", "knight"]


def test_cycle_rule_still_rejects_the_same_card_seconds_later():
    o = opp()
    assert seen_twice(o, [unit(1, "knight", 0.3, 0.3)], 10.0) == ["knight"]
    assert seen_twice(o, [unit(2, "knight", 0.75, 0.2)], 13.0) == []        # nouveau numéro, loin : même pose
    assert o.played == ["knight"]


# ---- 7. cartes ajoutées, versions évoluées, unités nées d'une autre ----
def test_evolution_unit_counts_as_its_base_card():
    o = opp()
    assert seen_twice(o, [unit(1, "knight-evolution", 0.3, 0.3)], 10.0) == ["knight"]
    assert seen_twice(o, [unit(2, "giant-hero", 0.7, 0.3)], 20.0) == ["giant"]
    assert o.elixir == pytest.approx(5 + 20.04 / 2.8 - 3 - 5)


def test_new_cards_have_their_cost():
    o = opp()
    assert seen_twice(o, [unit(1, "lava-hound", 0.3, 0.2)], 10.0) == ["lava-hound"]
    assert o.elixir == pytest.approx(10 - 7)            # carte lourde après un temps mort : il était plein
    assert UNIT2CARD["zappy"] == ("zappies", 4, 3) and UNIT2CARD["skeleton-dragon"] == ("skeleton-dragons", 4, 2)


def test_spawned_units_are_not_new_plays():
    o = opp()
    witch = unit(1, "witch", 0.3, 0.2)
    assert seen_twice(o, [witch], 10.0) == ["witch"]
    # squelettes de la Sorcière (elle est là, un peu plus loin) : pas la carte Squelettes
    assert seen_twice(o, [witch, *(unit(10 + i, "skeleton", 0.55 + 0.04 * i, 0.3) for i in range(3))], 14.0) == []
    # Géant gobelin : les gobelins à lance sur son dos naissent avec lui
    gg = [unit(20, "goblin-giant", 0.7, 0.25), unit(21, "spear-goblin", 0.68, 0.23), unit(22, "spear-goblin", 0.72, 0.23)]
    assert seen_twice(o, gg, 30.0) == ["goblin-giant"]
    # Pierre tombale : ses squelettes à côté d'elle
    tomb = unit(30, "tombstone", 0.5, 0.3)
    assert seen_twice(o, [tomb], 50.0) == ["tombstone"]
    assert seen_twice(o, [tomb, unit(31, "skeleton", 0.52, 0.33)], 53.0) == []
    assert o.played == ["witch", "goblin-giant", "tombstone"]


def test_unit2card_uses_exact_detector_classes_and_known_costs():
    pytest.importorskip("ultralytics")
    pt = Path(__file__).resolve().parents[1] / "models/yolo/clashai_yolo11s.pt"
    if not pt.exists():
        pytest.skip("modèle YOLO absent")
    os.environ.setdefault("YOLO_OFFLINE", "1")
    from ultralytics import YOLO
    classes = {n.rpartition("_")[0] for n in YOLO(str(pt)).names.values()}
    assert set(UNIT2CARD) - classes == set()
    db = json.loads((Path(__file__).resolve().parents[1] / "data/cards/cards_db.json").read_text(encoding="utf-8"))
    for name in ["lava-hound", "electro-giant", "goblin-giant", "sparky", "night-witch", "goblin-cage", "goblin-drill",
                 "skeleton-dragon", "electro-dragon", "mother-witch", "phoenix-big", "royal-ghost", "magic-archer",
                 "golden-knight", "archer-queen", "skeleton-king", "mighty-miner", "monk", "little-prince", "hunter",
                 "fisherman", "zappy", "flying-machine", "cannon-cart", "battle-healer", "bowler", "elixir-golem-big",
                 "heal-spirit"]:
        card, cost, _ = UNIT2CARD[name]
        assert db[card]["elixir"] == cost, name
    for v in [c for c in classes if c.endswith(("-evolution", "-hero"))]:
        base = v.rsplit("-", 1)[0]
        if base in UNIT2CARD:
            assert UNIT2CARD[v] == UNIT2CARD[base], v


# ---- 8. élixir ----
def test_elixir_bookkeeping():
    o = opp()
    assert o.elixir == 5.0
    o.update([], 2.8, H)
    assert o.elixir == pytest.approx(6.0)                                    # +1 toutes les 2,8 s
    gobs = [unit(1 + i, "goblin", 0.4 + 0.05 * i, 0.3) for i in range(3)]
    assert seen_twice(o, gobs, 5.6) == ["goblins"]                          # 3 gobelins = une carte, payée une fois
    assert o.elixir == pytest.approx(5 + (5.6 + DT) / 2.8 - 2)
    assert o.played == ["goblins"]
    o.update([], 60.0, H)
    assert o.elixir == 10.0                                                  # plafond
    o.update([], 120.0, H)
    o.elixir = 0.0
    o.update([], 121.4, H)
    assert o.elixir == pytest.approx(1.0)                                    # x2 après 120 s
    assert o.gained == pytest.approx(120.0 / 2.8 + 1.0)
