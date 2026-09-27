"""Camp et nom d'une unité votés sur toute sa vie."""
from clashai.identity import TrackIdentity

OWN = {"knight", "archer", "giant"}


def test_born_in_enemy_half_is_enemy_even_if_detector_says_ours():
    t = TrackIdentity(birth_y=0.2)
    for _ in range(3):
        t.observe("knight", 0.5, detector_enemy=False, color=0)
    assert t.enemy


def test_early_wrong_side_is_corrected_later():
    # avant : 8 premières images fausses = camp faux pour toute la vie
    t = TrackIdentity(birth_y=0.5)                      # née au pont : pas d'avis
    for _ in range(8):
        t.observe("giant", 0.9, detector_enemy=False, color=0)
    for _ in range(10):
        t.observe("giant", 0.9, detector_enemy=True, color=+1)   # badge rouge lisible
    assert t.enemy


def test_badge_colour_beats_birth_prior():
    t = TrackIdentity(birth_y=0.2)                      # ex. gobelins du Tonneau : ennemis, nés chez nous… ou l'inverse
    t.observe("goblin", 0.6, detector_enemy=True, color=-1)
    assert not t.enemy


def test_name_is_voted_over_the_track():
    t = TrackIdentity(birth_y=0.2)
    for n in ["knight"] * 5 + ["valkyrie"]:
        t.observe(n, 0.7, True, 1)
    assert t.name() == "knight"


def test_our_units_only_take_our_deck_names():
    t = TrackIdentity(birth_y=0.8)
    t.observe("musketeer", 0.8, False, -1)              # notre archère lue « mousquetaire »
    t.observe("archer", 0.5, False, -1)
    assert t.name(OWN) == "archer"
    assert t.name() == "musketeer"                       # sans restriction : le plus voté
