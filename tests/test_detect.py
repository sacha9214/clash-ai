"""Vitesse des unités (clashai/motion.py) et détecteurs : pytest -q tests/test_detect.py

Le test du vrai YOLO (CPU) est sauté si models/yolo/clashai_yolo11s.pt est absent ; detect_katacr.py est testé avec
des modules KataCR factices (ni KataCR ni ses modèles ici).
"""
import importlib
import math
import os
import random
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("YOLO_OFFLINE", "1")      # ultralytics : pas d'appel réseau pendant les tests

from clashai.motion import Motion, Tracks, feet  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
V = (45.0, -70.0)                                 # px/s : vitesse de référence des scénarios


def _times(duration: float, rng: random.Random, burst_every: float = 0.3) -> list[float]:
    """Instants d'images comme en match : boucle de 30 à 90 ms, plus des rafales de 8 images à 5 ms
    (pose d'une carte : agent._observe détecte chaque image reçue)."""
    ts, t, burst = [], 0.0, burst_every
    while t < duration:
        ts.append(t)
        if t >= burst:
            ts += [t + 0.005 * k for k in range(1, 9)]
            t += 0.04
            burst += burst_every
        t += rng.uniform(0.03, 0.09)
    return ts


def _err(v, ref=V) -> float:
    return math.hypot(v[0] - ref[0], v[1] - ref[1])


# ---- clashai/motion.py ----

def test_new_track_is_still_until_two_spaced_observations():
    m = Motion()
    assert m.update(10.0, 100, 200) == (0.0, 0.0)
    assert m.update(10.02, 101, 199) == (0.0, 0.0)          # fusionnée avec la 1re (< 30 ms)
    assert m.update(10.1, 104, 193) == (0.0, 0.0)           # écart trop court : ce serait du bruit
    m = Motion()
    m.update(10.0, 100, 200)
    vx, vy = m.update(10.2, 109, 186)
    assert vx == pytest.approx(45) and vy == pytest.approx(-70)


@pytest.mark.parametrize("seed", range(5))
def test_constant_motion_irregular_timestamps_and_bursts(seed):
    rng = random.Random(seed)
    m, t0 = Motion(), 5000.0 + seed                     # grands instants, comme time.perf_counter()
    for t in _times(3.0, rng):
        # boîtes en pixels entiers, comme celles du détecteur
        v = m.update(t0 + t, round(300 + V[0] * t), round(900 + V[1] * t))
        if t > 0.8:
            assert _err(v) < 0.03 * math.hypot(*V), (t, v)


@pytest.mark.parametrize("seed", range(5))
def test_jitter_robust(seed):
    rng = random.Random(seed)
    m, errs, naive = Motion(), [], []
    prev = None
    for t in _times(4.0, rng):
        x, y = 300 + V[0] * t + rng.gauss(0, 1.5), 900 + V[1] * t + rng.gauss(0, 1.5)
        v = m.update(t, round(x), round(y))
        if prev and t > 1.0:
            errs.append(_err(v))
            naive.append(_err(((x - prev[1]) / (t - prev[0]), (y - prev[2]) / (t - prev[0]))))
        prev = (t, x, y)
    assert max(naive) > 200                             # la gigue divisée par 5 ms : l'ancien problème
    assert sorted(errs)[len(errs) // 2] < 4.0           # médiane
    assert max(errs) < 9.0                              # < 0.3 case/s dans le pire cas


def test_gap_keeps_speed_then_expires():
    m = Motion()
    for k in range(20):                                 # 0 -> 0.76 s à 25 img/s
        m.update(0.04 * k, 300 + V[0] * 0.04 * k, 900 + V[1] * 0.04 * k)
    # masquée 0.6 s, elle avance toujours : sitôt revue, la vitesse est la bonne (pas de redémarrage à 0)
    t = 0.76 + 0.6
    assert _err(m.update(t, 300 + V[0] * t, 900 + V[1] * t)) < 1.0
    # trou de plus de MAX_GAP_S : on repart de zéro (vitesse périmée), puis on retrouve la vitesse
    t += 1.5
    assert m.update(t, 300 + V[0] * t, 900 + V[1] * t) == (0.0, 0.0)
    for k in range(1, 6):
        v = m.update(t + 0.04 * k, 300 + V[0] * (t + 0.04 * k), 900 + V[1] * (t + 0.04 * k))
    assert _err(v) < 1.0


def test_stop_is_seen_within_the_window():
    m, t = Motion(), 0.0
    for k in range(25):
        t = 0.04 * k
        m.update(t, 300 + V[0] * t, 900 + V[1] * t)
    x, y = 300 + V[0] * t, 900 + V[1] * t                # l'unité s'arrête (elle attaque)
    t_stop = t
    while t < t_stop + Motion.WINDOW_S + 0.1:
        t += 0.04
        v = m.update(t, x, y)
    assert v == pytest.approx((0.0, 0.0), abs=1e-6)


class _U:
    """Unité minimale : ce que Tracks lit (track_id, box) et écrit (vel)."""
    def __init__(self, track_id, box):
        self.track_id, self.box, self.vel = track_id, box, (0.0, 0.0)


def test_tracks_bottom_center_untracked_and_forget():
    tr = Tracks()
    a = _U(3, (100, 200, 130, 240))
    b = _U(-1, (0, 0, 10, 10))
    tr.observe([a, b], 0.0)
    # la boîte grandit vers le haut (animation), les pieds ne bougent pas : vitesse nulle
    a2 = _U(3, (100, 170, 130, 240))
    tr.observe([a2], 0.3)
    assert a2.vel == (0.0, 0.0) and b.vel == (0.0, 0.0) and -1 not in tr
    a3 = _U(3, (115, 170, 145, 270))
    tr.observe([a3], 0.6)
    assert a3.vel[0] > 0 and a3.vel[1] > 0              # x vers la droite, y vers le bas
    assert feet(a3.box) == (130.0, 270.0)
    tr.forget(0.6 + Tracks.KEEP_S - 0.01)
    assert 3 in tr
    tr.forget(0.6 + Tracks.KEEP_S + 0.01)
    assert 3 not in tr


# ---- detect_yolo.py : le vrai Detector, avec un modèle factice qui rejoue un scénario ----

FRAME = (1280, 578)          # flux du téléphone (h, w)


def _stream_scale():
    from clashai.detect_yolo import ARENA, ARENA_SIZE
    h, w = FRAME
    x0, x1 = int(ARENA[0] * w), int((ARENA[0] + ARENA[2]) * w)
    y0, y1 = int(ARENA[1] * h), int((ARENA[1] + ARENA[3]) * h)
    return (x1 - x0) / ARENA_SIZE[0], (y1 - y0) / ARENA_SIZE[1]


def _knight(t, vc):
    """Boîte (arène 568x896) d'un chevalier ennemi qui avance à vc px/s."""
    x, y = 250 + vc[0] * t, 300 + vc[1] * t
    return x, y - 40, x + 30, y


@pytest.fixture
def fake_yolo(monkeypatch):
    pytest.importorskip("ultralytics")
    torch = pytest.importorskip("torch")
    import clashai.detect_yolo as dy
    script = {"rows": []}                # (suivi, x0, y0, x1, y1, classe, confiance) de la prochaine image

    class Boxes:
        def __init__(self, rows):
            b = torch.tensor(rows, dtype=torch.float32).reshape(-1, 7)
            self.id = b[:, 0] if len(rows) else None
            self.xyxy, self.cls, self.conf = b[:, 1:5], b[:, 5], b[:, 6]

        def __len__(self):
            return len(self.xyxy)

    class FakeYOLO:
        def __init__(self, *a, **k):
            pass

        def track(self, crop, **kw):
            return [SimpleNamespace(boxes=Boxes(script["rows"]), names={0: "knight_1", 1: "archer_0"})]
        predict = track

    monkeypatch.setattr(dy, "YOLO", FakeYOLO)
    return dy, script


def test_yolo_detector_velocity_coasting_and_trails(fake_yolo):
    dy, script = fake_yolo
    det = dy.Detector(device="cpu", track=True)
    frame = np.zeros((*FRAME, 3), np.uint8)
    vc = (20.0, 50.0)                                    # px/s dans l'arène recadrée
    sx, sy = _stream_scale()
    want = (vc[0] * sx, vc[1] * sy)                      # px/s dans l'image du flux
    rng, t0 = random.Random(1), 100.0
    ts = _times(1.5, rng)
    for t in ts:
        script["rows"] = [(7, *_knight(t, vc), 0, 0.9)]
        units = det(frame, t=t0 + t)
        assert len(units) == 1 and isinstance(units[0], dy.Unit)
        u = units[0]
        assert u.track_id == 7 and u.enemy and not u.coasted
        if t > 0.8:
            assert _err(u.vel, want) < 0.04 * math.hypot(*want), (t, u.vel)
    t_last, box_last, vel_last = ts[-1], u.box, u.vel
    # ratée 3 images : gardée de mémoire, avec sa dernière vitesse et sa trace
    script["rows"] = []
    for k in (1, 2, 3):
        t = t_last + 0.1 * k
        (u,) = det(frame, t=t0 + t)
        assert u.coasted and u.vel == vel_last and 7 in det.trails
        assert u.box[3] - box_last[3] == pytest.approx(vel_last[1] * 0.1 * k, abs=1)
    dy.draw(frame, [u], det.trails)
    # revue : la vitesse continue (pas de redémarrage à 0)
    for k in (4, 5):
        t = t_last + 0.1 * k
        script["rows"] = [(7, *_knight(t, vc), 0, 0.9)]
        (u,) = det(frame, t=t0 + t)
        assert not u.coasted and _err(u.vel, want) < 0.06 * math.hypot(*want)
    # plus prévue au-delà de COAST_S, mais trace gardée tant qu'elle est en mémoire ; puis tout est oublié
    script["rows"] = []
    assert det(frame, t=t0 + t + 1.0) == [] and 7 in det.trails
    assert det(frame, t=t0 + t + 2.5) == []
    assert 7 not in det.trails and 7 not in det.memory and 7 not in det.motion
    det.reset()
    script["rows"] = [(8, *_knight(0, vc), 0, 0.9)]
    assert det(frame, t=t0 + 10)[0].vel == (0.0, 0.0)   # nouveau suivi


def test_yolo_untracked_has_zero_velocity(fake_yolo):
    dy, script = fake_yolo
    det = dy.Detector(device="cpu", track=False)
    script["rows"] = [(-1, *_knight(0, (0, 0)), 1, 0.9)]
    for t in (0.0, 0.5):
        (u,) = det.on_arena(np.zeros((896, 568, 3), np.uint8), t=t)
        assert u.track_id == -1 and u.vel == (0.0, 0.0)


# ---- detect_katacr.py : modules KataCR factices (ultralytics 8.1 et KataCR absents ici) ----

@pytest.fixture
def katacr(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    names = {0: "knight", 1: "bar"}
    script = {"rows": []}                # (x0, y0, x1, y1, confiance, classe, camp) de la prochaine image

    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, m)

    class CRResults:
        def __init__(self, orig_img, path, names, boxes):
            self.boxes = boxes

        def get_data(self):
            return self.boxes.numpy()

    def start(pred, persist=False):
        pred.tracker = SimpleNamespace(reset=lambda: None)

    def post_end(pred, persist=False):   # ByteTrack factice : suivi = 1 + rang de la boîte
        b = pred.result.boxes
        ids = torch.arange(1, len(b) + 1, dtype=b.dtype)[:, None]
        pred.result.boxes = torch.cat([b[:, :4], ids, b[:, 4:]], 1)

    class YOLO_CR:
        def __init__(self, path):
            pass

        def predict(self, crop, **kw):
            return [SimpleNamespace(orig_boxes=torch.tensor(script["rows"], dtype=torch.float32).reshape(-1, 7),
                                    names=names)]

    for pkg in ("katacr", "katacr.constants", "katacr.yolov8"):
        mod(pkg)
    mod("katacr.constants.label_list", idx2unit=names, unit2idx={v: k for k, v in names.items()})
    mod("katacr.yolov8.custom_result", CRResults=CRResults)
    mod("katacr.yolov8.custom_trackers", cr_on_predict_start=start, cr_on_predict_postprocess_end=post_end)
    mod("katacr.yolov8.train", YOLO_CR=YOLO_CR)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    sys.modules.pop("clashai.detect_katacr", None)
    yield importlib.import_module("clashai.detect_katacr"), script
    sys.modules.pop("clashai.detect_katacr", None)
    sys.modules["clashai"].__dict__.pop("detect_katacr", None)


def test_katacr_detector_velocity_and_trails(katacr):
    dk, script = katacr
    det = dk.Detector(device="cpu", track=True, weights=["factice.pt"])
    frame = np.zeros((*FRAME, 3), np.uint8)
    vc = (20.0, 50.0)
    sx, sy = _stream_scale()
    want = (vc[0] * sx, vc[1] * sy)
    ts = _times(1.5, random.Random(2))
    for t in ts:
        script["rows"] = [(*_knight(t, vc), 0.9, 0, 1)]
        (u,) = det(frame, t=t)
        assert isinstance(u, dk.Unit) and u.track_id == 1 and u.enemy
        if t > 0.8:
            assert _err(u.vel, want) < 0.04 * math.hypot(*want), (t, u.vel)
    # ratée 0.3 s : trace et vitesse gardées ; revue : la vitesse continue
    script["rows"] = []
    assert det(frame, t=ts[-1] + 0.3) == [] and 1 in det.trails
    t = ts[-1] + 0.4
    script["rows"] = [(*_knight(t, vc), 0.9, 0, 1)]
    (u,) = det(frame, t=t)
    assert _err(u.vel, want) < 0.06 * math.hypot(*want)
    dk.draw(frame, [u], det.trails)
    script["rows"] = []
    det(frame, t=t + 2.5)
    assert 1 not in det.trails and 1 not in det.motion


# ---- smoke test : le vrai YOLO sur CPU ----

def test_yolo_real_model_smoke():
    if not (ROOT / "models/yolo/clashai_yolo11s.pt").exists():
        pytest.skip("modèle YOLO absent (models/yolo/clashai_yolo11s.pt)")
    pytest.importorskip("ultralytics")
    from clashai.detect_yolo import Detector, Unit, draw
    det = Detector(device="cpu", track=True)
    rng = np.random.default_rng(0)
    for k in range(4):
        img = np.full((*FRAME, 3), (60, 140, 90), np.uint8)             # « herbe »
        img[:] = np.clip(img + rng.integers(-20, 20, img.shape), 0, 255).astype(np.uint8)
        for j in range(3):                                                # quelques taches qui bougent
            x, y = 100 + 150 * j + 6 * k, 300 + 200 * j + 10 * k
            img[y:y + 40, x:x + 30] = (40, 40, 220) if j % 2 else (220, 120, 40)
        units = det(img)
        assert isinstance(units, list) and all(isinstance(u, Unit) for u in units)
        for u in units:
            assert isinstance(u.vel, tuple) and len(u.vel) == 2
            assert all(isinstance(c, float) and math.isfinite(c) for c in u.vel)
        draw(img, units, det.trails)
