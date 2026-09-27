"""Boucle de combat, lecture du résultat, tours, connexion : sans téléphone (images et appareil factices)."""
import importlib
import importlib.util
import os
import time

import numpy as np
import pytest

from clashai import battle as B
from clashai.towers import MatchState

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W, H = 578, 1280


def _autoplay():
    spec = importlib.util.spec_from_file_location("autoplay", os.path.join(ROOT, "scripts/autoplay.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # main() n'est pas lancé à l'import
    return mod


# ---- résultat du combat (couronnes dorées) ----
def _end_screen(ours: int, theirs: int) -> np.ndarray:
    img = np.full((H, W, 3), 30, np.uint8)
    gold = (40, 180, 240)                 # BGR
    for i in range(ours):                 # bandeau bleu (bas), 11 % de haut
        x = 120 + 110 * i
        img[600:640, x:x + 40] = gold
    for i in range(theirs):               # bandeau rouge (haut), 15 % de haut
        x = 120 + 110 * i
        img[300:340, x:x + 40] = gold
    return img


@pytest.mark.parametrize("ours,theirs,expected", [
    (0, 0, "draw"), (1, 1, "draw"), (2, 2, "draw"), (3, 3, "draw"),     # égalités : avant, 1-1 à 3-3 = « win »
    (1, 0, "win"), (3, 1, "win"), (0, 1, "loss"), (1, 2, "loss"),
])
def test_result_of_counts_crowns(ours, theirs, expected):
    assert _autoplay().result_of(_end_screen(ours, theirs)) == expected


# ---- boucle de combat : flux coupé ou figé ----
def _battle_image() -> np.ndarray:
    img = np.zeros((H, W, 3), np.uint8)
    y = int(B.ELIXIR_Y * H)
    img[y - 3:y + 4, :] = (200, 50, 200)  # barre d'élixir rose + goutte magenta : « en combat »
    assert B.in_battle(img)
    return img


class _FakeDevice:
    def __init__(self, img, alive=True):
        self.img, self.alive, self.n = img, alive, 7

    def wait_frame(self, timeout=1.0, after=None):
        time.sleep(0.01)
        return False                      # plus jamais de nouvelle image

    def frame(self):
        return self.img, time.perf_counter(), self.n


class _FakeDet:
    tracker = None
    trails = {}


def _agent(tmp_path, think_calls):
    from clashai.agent import Agent
    a = Agent.__new__(Agent)              # sans charger le détecteur
    a.out, a.show, a.det = str(tmp_path / "games"), False, _FakeDet()

    def think(img, now, fps):
        think_calls.append(now)
        return [], None, [], [None] * 4, 5.0
    a.think = think
    return a


def test_dead_stream_stops_the_battle(tmp_path):
    from clashai.agent import StreamLost
    calls = []
    a = _agent(tmp_path, calls)
    t0 = time.time()
    with pytest.raises(StreamLost):
        a.play_battle(_FakeDevice(_battle_image(), alive=False), "g1", {"placement_model": False})
    assert time.time() - t0 < 2 and len(calls) == 1      # une seule analyse de l'unique image


def test_frozen_stream_stops_after_stall_delay(tmp_path):
    from clashai.agent import StreamLost
    calls = []
    a = _agent(tmp_path, calls)
    a.STALL_S = 0.3
    with pytest.raises(StreamLost):
        a.play_battle(_FakeDevice(_battle_image(), alive=True), "g2", {"placement_model": False})
    assert len(calls) == 1                # avant : ~45 000 analyses/s de la même image, sans fin


# ---- modèle adverse pendant la pose d'une carte ----
def test_heavy_card_seen_during_a_play_triggers_punish(tmp_path):
    from clashai.brain import Brain
    a = _agent(tmp_path, [])
    a.brain, a.opp_log = Brain({"placement_model": False}), []

    class _Opp:
        elixir = 3.0

        def update(self, units, now, h):
            return ["golem"]
    a.opp = _Opp()
    a._opp_update([], 123.0, H)
    assert a.brain.opp_heavy_t == 123.0 and a.opp_log[0]["card"] == "golem"


# ---- tours ----
class _U:
    def __init__(self, name, cx, cy):
        self.name, self.center = name, (cx, cy)


def test_cannoneer_towers_are_alive():
    t0 = time.time()                      # même horloge que le match (tours vues pour la dernière fois à la création)
    m = MatchState(start=t0)
    img = np.zeros((H, W, 3), np.uint8)
    for t in range(0, 10):
        m.update(img, [_U("cannoneer-tower", 118, 272), _U("dagger-duchess-tower", 458, 272)], now=t0 + t)
    assert m.enemy_alive == {0: True, 1: True}      # avant : les deux « détruites » au bout de 3 s


def _bar_image(frac_left: float | None) -> np.ndarray:
    img = np.zeros((H, W, 3), np.uint8)
    if frac_left is not None:
        x0, x1, y = 0.168 * W, 0.285 * W, int(0.592 * H)
        img[y - 1:y + 2, int(x0):int(x0 + (x1 - x0) * frac_left)] = (230, 150, 20)   # bleu (BGR)
    return img


def test_destroyed_tower_bar_goes_to_zero():
    m = MatchState(start=0.0)
    for t in range(5):
        m.update(_bar_image(0.2), [], now=float(t))
    assert m.our_hp[0] == pytest.approx(0.2, abs=0.05)
    m.update(_bar_image(None), [], now=8.0)
    assert m.our_hp[0] == pytest.approx(0.2, abs=0.05)   # cachée 4 s : peut-être une unité devant
    m.update(_bar_image(None), [], now=11.0)
    assert m.our_hp[0] == 0.0                             # basse puis disparue 7 s : détruite


def test_healthy_hidden_bar_is_not_destroyed():
    m = MatchState(start=0.0)
    m.update(_bar_image(0.9), [], now=0.0)
    m.update(_bar_image(None), [], now=30.0)
    assert m.our_hp[0] > 0.5


# ---- connexion ----
def test_device_import_without_scrcpy(monkeypatch):
    monkeypatch.delenv("SCRCPY_SERVER_PATH", raising=False)
    import clashai.device as D
    D = importlib.reload(D)               # avant : IndexError à l'import quand scrcpy-server est introuvable
    assert isinstance(D.SERVER_LOCAL, str)


# ---- PV des tours ennemies (barre rouge au-dessus de chaque tour, calibrée en début de match) ----
TOWER = (80, 240, 160, 320)                                   # boîte de la tour ennemie gauche (détecteur)


class _Tower:
    name, box = "queen-tower", TOWER
    center = ((TOWER[0] + TOWER[2]) // 2, (TOWER[1] + TOWER[3]) // 2)


def _tower_frame(frac):
    img = np.zeros((H, W, 3), np.uint8)
    img[222:228, 85:85 + int(70 * frac)] = (30, 30, 230)       # barre rouge (BGR) au-dessus de la tour
    return img


def test_enemy_tower_hp_self_calibrates_and_ignores_occlusion():
    m = MatchState(start=0.0)
    for t in range(8):                                          # 20 premières s : tour intacte = barre pleine
        m.update(_tower_frame(1.0), [_Tower()], now=1.0 + t)
    assert m.enemy_hp_ok
    for t in range(9):
        m.update(_tower_frame(0.3), [_Tower()], now=30.0 + t)
    assert m.enemy_hp[0] == pytest.approx(0.3, abs=0.05)
    for t in range(3):                                          # 3 images masquées : la médiane tient
        m.update(_tower_frame(0.0), [_Tower()], now=40.0 + t)
    assert m.enemy_hp[0] == pytest.approx(0.3, abs=0.05)


def test_enemy_tower_hp_off_when_no_clean_full_bar():
    m = MatchState(start=0.0)
    rng = np.random.default_rng(0)
    for t in range(8):                                          # longueurs incohérentes : lecture non fiable
        m.update(_tower_frame(float(rng.uniform(0.2, 1.0))), [_Tower()], now=1.0 + t)
    assert not m.enemy_hp_ok
    m = MatchState(start=0.0)
    for t in range(8):                                          # aucune barre visible
        m.update(_tower_frame(0.0), [_Tower()], now=1.0 + t)
    assert not m.enemy_hp_ok
