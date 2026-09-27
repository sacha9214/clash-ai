"""Outils de fin de session : scripts/match_report.py (page HTML autonome) et scripts/pack_samples.py (échantillons à pousser).
Tout est écrit dans tmp_path : les vrais journaux et learning/ ne sont jamais touchés."""
import importlib.util
import json
import re
import shutil
import tarfile
from html.parser import HTMLParser
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MR, PS = _script("match_report"), _script("pack_samples")
VOID = {"meta", "br", "hr", "img", "input", "link", "col", "wbr", "area", "base", "source", "track", "embed", "param"}


class Balanced(HTMLParser):
    """Vérifie que chaque balise fermée est la dernière ouverte (page bien formée)."""

    def __init__(self):
        super().__init__()
        self.stack, self.errors = [], []

    def handle_starttag(self, tag, attrs):
        if tag not in VOID:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        pass

    def handle_endtag(self, tag):
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"</{tag}> alors que <{self.stack[-1] if self.stack else '-'}> est ouverte")
        else:
            self.stack.pop()


def check_page(text: str):
    p = Balanced()
    p.feed(text)
    p.close()
    assert not p.errors, p.errors[:5]
    assert not p.stack, p.stack
    assert not re.search(r"https?:|//[a-z0-9.-]+\.[a-z]{2,}/", text, re.I)       # aucune ressource externe
    assert not re.search(r"<(link|img|iframe)\b|@import|\bsrc\s*=", text, re.I)
    data = re.search(r'<script type="application/json" id="report-data">(.*?)</script>', text, re.S)
    return json.loads(data.group(1))


def synthetic_game(games: Path, gid="20260927-120000") -> str:
    """Match récent minimal : opp_elixir et trade dans les décisions, trade_balance dans le journal."""
    d = games / gid
    d.mkdir(parents=True)
    t = 1790500000.0
    rows = [{"t": t, "card": "giant", "x": 0.2, "y": 0.7, "tile": [3, 28], "reason": "attaque : Géant (avance 3)",
             "ok": True, "play_ms": 140, "elixir": 9.2, "hand": ["giant", "knight", "arrows", "fireball"],
             "opp_elixir": 6.1, "trade": None, "units": [["king-tower", True, [289, 176]]]},
            {"t": t + 6, "card": "arrows", "x": 0.5, "y": 0.4, "tile": [9, 12], "reason": "arrows sur un groupe de 3 (4.0 élixir détruits)",
             "ok": True, "play_ms": 150, "elixir": 5.1, "hand": ["knight", "arrows", "fireball", "archers"],
             "opp_elixir": 2.3, "trade": 1.5, "units": [["minion", True, [280, 500]]]},
            {"t": t + 9, "card": "knight", "x": 0.7, "y": 0.6, "tile": [13, 22], "reason": "défense : hog (tank) -> knight",
             "ok": False, "play_ms": 900, "elixir": 3.4, "hand": ["knight", "fireball", "archers", "giant"],
             "opp_elixir": 4.0, "trade": -0.5, "units": []}]
    (d / "decisions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (d / "opponent.json").write_text(json.dumps({"played": [{"t": t + 2, "card": "minions", "elixir_after": 3.1},
                                                            {"t": t + 8, "card": "hog-rider", "elixir_after": 0.5}],
                                                 "deck": ["minions", "hog-rider"]}), encoding="utf-8")
    (d / "spells.jsonl").write_text("", encoding="utf-8")
    return json.dumps({"game": gid, "played": 2, "refused": 1, "params": {"giant_elixir": 8}, "trade_balance": 1.0,
                       "enemy_deck": ["minions", "hog-rider"], "result": "win", "counted": True})


def test_report_synthetic_new_fields(tmp_path):
    games = tmp_path / "runs/games"
    line = synthetic_game(games)
    (games / "journal.jsonl").write_text(line + "\n", encoding="utf-8")
    stats = tmp_path / "stats.json"
    stats.write_text(json.dumps({"a": {"params": {"giant_elixir": 8}, "wins": 3, "losses": 1}}), encoding="utf-8")
    out = MR.main(["--games", str(games), "--stats", str(stats), "--out", str(tmp_path / "report.html")])
    text = out.read_text(encoding="utf-8")
    data = check_page(text)
    assert "20260927-120000" in text and 'id="g-20260927-120000"' in text
    assert "+1.5" in text and "−0.5" in text and "refusé" in text        # échanges et coup refusé dans la liste
    assert "5.1 / 2.3" in text                                            # élixir adverse lu en direct (opp_elixir)
    assert 'id="bandit"' in text and "giant_elixir" in text and "meilleure" in text
    g = data["games"]["20260927-120000"]
    assert g["him"] and g["us"] and len(g["ev"]) == 5                     # 3 décisions + 2 cartes adverses
    assert all(0 <= y <= 10 for _, y in g["us"] + g["him"])
    assert [e[1] for e in g["ev"]] == [0, 1, 0, 1, 0]                     # chronologique, nous / lui
    # la courbe adverse passe par l'estimation en direct (opp_elixir) au moment de nos coups
    assert [0.0, 6.1] in g["him"] and [6.0, 2.3] in g["him"]


@pytest.fixture(scope="module")
def real_games(tmp_path_factory):
    """Les 52 vrais matchs du dépôt (learning/games_logs.tgz), extraits à part : partout, CI comprise."""
    tgz = ROOT / "learning/games_logs.tgz"
    if not tgz.exists():
        pytest.skip("learning/games_logs.tgz absent")
    d = tmp_path_factory.mktemp("logs")
    with tarfile.open(tgz) as t:
        t.extractall(d, filter="data")
    return d / "runs/games"


def test_report_real_games(tmp_path, real_games):
    REAL_GAMES = real_games
    ids = sorted(p.name for p in REAL_GAMES.iterdir() if (p / "decisions.jsonl").exists())
    ids = [ids[0], ids[len(ids) // 2], ids[-1]]                           # ancien (sans opponent.json), milieu, récent
    games = tmp_path / "runs/games"
    for gid in ids:
        shutil.copytree(REAL_GAMES / gid, games / gid)
    journal = [l for l in (ROOT / "learning/journal.jsonl").read_text(encoding="utf-8").splitlines()
               if json.loads(l)["game"] in ids]
    (games / "journal.jsonl").write_text("\n".join(journal) + "\n", encoding="utf-8")
    out = MR.main(["--games", str(games), "--stats", str(ROOT / "learning/strategy_stats.json"),
                   "--out", str(tmp_path / "r.html")])
    text = out.read_text(encoding="utf-8")
    data = check_page(text)
    for gid in ids:
        assert f'id="g-{gid}"' in text
    assert set(data["games"]) == set(ids)
    assert len(data["win"]["games"]) == 3
    for k in ("giant_elixir", "giant_spot", "defend_line"):
        assert f"<h3>{k}</h3>" in text
    assert text.count("meilleure") >= len(MR.ST.SPACE)                   # une meilleure valeur par paramètre

    # --game : un seul match en détail, la vue d'ensemble garde les trois
    one = MR.main(["--games", str(games), "--game", ids[-1], "--out", str(tmp_path / "one.html")]).read_text(encoding="utf-8")
    check_page(one)
    assert f'id="g-{ids[-1]}"' in one and f'id="g-{ids[0]}"' not in one
    with pytest.raises(SystemExit):
        MR.main(["--games", str(games), "--game", "19990101-000000", "--out", str(tmp_path / "x.html")])


def test_curve_regen_and_cap():
    pts, before = MR._curve([(10.0, None, 1.0)], 0.0, 200.0)
    assert before == [pytest.approx(5 + 10 / 2.8)]
    assert max(y for _, y in pts) == 10.0                                 # plafond
    assert pts[-1] == (200.0, 10.0)
    fast = MR._regen(120.0, 0.0, 121.4, 120.0)                              # x2 après 2 min : 1 élixir en 1,4 s
    assert fast[-1][1] == pytest.approx(1.0)


# --- pack_samples ---------------------------------------------------------------------------------------------

@pytest.fixture
def tree(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    rng = np.random.default_rng(0)
    for g in ("20260927-100000", "20260927-110000", "20260927-120000"):
        (runs / "capture" / g).mkdir(parents=True)
        for i in range(6):
            img = rng.integers(0, 255, (1280, 578, 3), dtype=np.uint8)
            cv2.imwrite(str(runs / "capture" / g / f"{i:07d}.jpg"), img)
    (runs / "capture" / "20260927-120000" / "broken.jpg").write_bytes(b"pas une image")
    v = runs / "videos"
    v.mkdir(parents=True)
    (v / "abc.jsonl").write_text('{"battle": 0}\n', encoding="utf-8")
    (v / "abc.battles.json").write_text('{"video": "abc"}', encoding="utf-8")
    (v / "abc.raw.jsonl").write_text("x" * 100, encoding="utf-8")
    (v / "big.jsonl").write_text("x" * 5000, encoding="utf-8")
    monkeypatch.setattr(PS, "ROOT", tmp_path)
    monkeypatch.setattr(PS, "CAPTURE", runs / "capture")
    monkeypatch.setattr(PS, "VIDEOS", v)
    monkeypatch.setattr(PS, "OUT", tmp_path / "learning/samples")
    monkeypatch.setattr(PS, "MAX_FILE", 4000)
    return tmp_path


def test_pack_samples(tree, capsys):
    out = tree / "learning/samples"
    (out / "frames").mkdir(parents=True)
    (out / "frames/old.jpg").write_bytes(b"ancien tirage")                 # refait à chaque fois
    m = PS.main(["--frames", "9", "--seed", "1"])
    frames = sorted((out / "frames").glob("*.jpg"))
    assert len(frames) == 9 and m["frames"]["count"] == 9 and m["frames"]["available"] == 19
    assert {f.name.split("_")[0] for f in frames} == {"20260927-100000", "20260927-110000", "20260927-120000"}
    for f in frames:                                                       # réduites : côté long 960
        img = cv2.imread(str(f))
        assert max(img.shape[:2]) == 960
    assert sorted(p.name for p in (out / "videos").iterdir()) == ["abc.battles.json", "abc.jsonl"]
    man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert man["frames"]["count"] == 9 and man["videos"]["count"] == 2
    assert [s["source"] for s in man["videos"]["skipped"]] == ["runs/videos/big.jsonl"]
    assert {f["path"] for f in man["files"]} == {f"frames/{p.name}" for p in frames} | {"videos/abc.jsonl", "videos/abc.battles.json"}
    assert all(f["source"].startswith("runs/") and "\\" not in f["source"] for f in man["files"])
    assert man["total_bytes"] == sum(f["bytes"] for f in man["files"])
    printed = capsys.readouterr().out
    assert "git add learning/samples" in printed and "git push" in printed and "git commit -m" in printed


def test_pack_samples_nothing_found(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(PS, "ROOT", tmp_path)
    monkeypatch.setattr(PS, "CAPTURE", tmp_path / "runs/capture")
    monkeypatch.setattr(PS, "VIDEOS", tmp_path / "runs/videos")
    monkeypatch.setattr(PS, "OUT", tmp_path / "learning/samples")
    m = PS.main([])
    assert m["files"] == [] and not (tmp_path / "learning").exists()     # rien créé, rien effacé
    assert "rien à envoyer" in capsys.readouterr().out


def test_pack_samples_size_cap(tree, capsys):
    one = len(PS._encode(next((tree / "runs/capture").rglob("0000000.jpg")), 960, 85))
    cap_mb = (PS.MANIFEST_ROOM + 3.5 * one) / 1024 / 1024                 # place pour 3 images environ
    m = PS.main(["--frames", "12", "--max-mb", f"{cap_mb:.6f}", "--seed", "0"])
    out = tree / "learning/samples"
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    assert m["frames"]["capped"] and 1 <= m["frames"]["count"] < 12
    assert total <= cap_mb * 1024 * 1024
    assert len(list((out / "frames").glob("*.jpg"))) == m["frames"]["count"]
    assert "plafond" in capsys.readouterr().out
