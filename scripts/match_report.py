"""Bilan d'une session de matchs : une page HTML autonome (CSS, SVG et JS en ligne, aucun accès réseau)
à ouvrir après les combats pour comprendre en 30 s pourquoi on a gagné ou perdu.

  - Vue d'ensemble : victoires / défaites, taux glissant, victoires vs défaites, cartes adverses difficiles, une ligne par match.
  - Chaque match (le plus récent en premier) : élixir des deux camps au fil du temps (le nôtre lu à l'écran, le sien
    reconstitué depuis opponent.json ou opp_elixir), cartes posées, liste des décisions, décompte par type.
  - Ce que le bandit a appris : par paramètre, victoires / défaites et proba a posteriori (clashai.strategy.param_table).

  python scripts/match_report.py                                   # runs/games -> runs/report.html
  python scripts/match_report.py --game 20260927-083411            # un seul match en détail
  python scripts/match_report.py --games D:/logs/runs/games --out D:/report.html
"""
from __future__ import annotations

import argparse
import collections
import html
import json
import re
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from clashai import strategy as ST  # noqa: E402
from clashai.cards import BENCH, DECK  # noqa: E402
from clashai.opponent import CARD_COST  # noqa: E402

REGEN, DOUBLE_AFTER, MAX_ELIXIR = 1 / 2.8, 120, 10.0
COST = {**CARD_COST, **{c.name: c.cost for c in [*DECK.values(), *BENCH.values()]}}
ROLL = 10   # fenêtre du taux de victoire glissant
RESULT = {"win": ("▲", "victoire"), "loss": ("▼", "défaite"), "draw": ("=", "égalité")}
esc = html.escape


def _jsonl(path: Path) -> list[dict]:
    """Lignes JSON d'un fichier ; une ligne tronquée (match coupé pendant l'écriture) est ignorée."""
    rows = []
    if path and path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    return rows


def _date(gid: str) -> str:
    m = re.match(r"\d{4}(\d\d)(\d\d)-(\d\d)(\d\d)", gid)
    return f"{m[2]}/{m[1]} {m[3]}:{m[4]}" if m else gid


def _mmss(s: float) -> str:
    return ("-" if s < 0 else "") + f"{int(abs(s)) // 60}:{int(abs(s)) % 60:02d}"


def _signed(v) -> str:
    return "" if v is None else f"{v:+.1f}".replace("-", "−")


def _pct(v: float) -> str:
    return f"{round(v)}\u202f%"


def _kind(reason: str) -> str:
    """Type de décision : « défense », « attaque », « soutien »… ; « fireball sur un groupe de 3 » -> « sort »."""
    if " : " in reason:
        return reason.split(" : ")[0].strip()
    return "sort" if " sur " in reason else (reason.strip() or "?")


def _regen(t: float, v: float, until: float, t2: float) -> list[tuple[float, float]]:
    """Recharge de t à until (1 élixir / 2,8 s, x2 après t2, plafond 10) : points de la courbe, coudes compris."""
    pts = []
    for edge in ([t2] if t < t2 < until else []) + [until]:
        rate = REGEN * (2 if t >= t2 else 1)
        full = t + (MAX_ELIXIR - v) / rate
        if t < full < edge:
            pts.append((full, MAX_ELIXIR))
        v, t = min(MAX_ELIXIR, v + (edge - t) * rate), edge
        pts.append((t, v))
    return pts


def _curve(anchors: list, t0: float, end: float, regen: bool = True) -> tuple[list, list]:
    """Élixir au fil du temps : 5 au départ, puis ancres (t, valeur lue ou None, valeur après la carte) triées ;
    entre deux ancres la barre se recharge (regen=False : segment droit entre deux lectures, notre barre).
    Renvoie les points et la valeur juste avant chaque ancre."""
    t, v, pts, before = t0, 5.0, [(t0, 5.0)], []
    for ta, seen, after in anchors:
        if regen and ta > t:
            pts += _regen(t, v, ta, t0 + DOUBLE_AFTER)
            v = pts[-1][1]
        t, v0 = max(t, ta), v if seen is None else seen
        v = v0 if after is None else after
        pts += [(ta, v0), (ta, v)]
        before.append(v0)
    if regen and end > t:
        pts += _regen(t, v, end, t0 + DOUBLE_AFTER)
    return pts, before


def _at(pts: list, x: float) -> float:
    """Valeur de la courbe à l'instant x (avant un éventuel saut à cet instant)."""
    for (xa, ya), (xb, yb) in zip(pts, pts[1:]):
        if xa <= x <= xb:
            return ya if xb == xa or x == xa else ya + (yb - ya) * (x - xa) / (xb - xa)
    return pts[-1][1]


def load_game(gid: str, folder: Path | None, j: dict) -> dict:
    """Un match : journal (résultat, réglages, deck adverse) + decisions.jsonl + opponent.json."""
    rows = [r for r in (_jsonl(folder / "decisions.jsonl") if folder else []) if "t" in r and "card" in r]
    for r in rows:
        r.setdefault("reason", ""), r.setdefault("ok", True), r.setdefault("elixir", None)
    opp = {}
    if folder and (folder / "opponent.json").exists():
        try:
            opp = json.loads((folder / "opponent.json").read_text(encoding="utf-8"))
        except ValueError:
            pass
    his = sorted((e for e in opp.get("played") or [] if "t" in e and "card" in e), key=lambda e: e["t"])
    trades = [r["trade"] for r in rows if r["ok"] and r.get("trade") is not None]
    trade = j.get("trade_balance", round(sum(trades), 1) if trades else None)
    g = {"id": gid, "date": _date(gid), "result": j.get("result", "?"), "counted": j.get("counted", True),
         "rows": rows, "his": his, "deck": j.get("enemy_deck") or opp.get("deck") or [],
         "params": j.get("params") or {}, "refused": j.get("refused", sum(not r["ok"] for r in rows)),
         "played": j.get("played", sum(r["ok"] for r in rows)), "trade": trade, "chart": None}
    if not rows and not his:
        return g
    tf = rows[0]["t"] if rows else his[0]["t"]                        # origine des temps : première décision
    t0 = min([r["t"] for r in rows[:1]] + [e["t"] for e in his[:1]]) - 3   # début estimé du combat (replay_logs)
    end = max([r["t"] for r in rows[-1:]] + [e["t"] for e in his[-1:]]) + 2
    us, us_before = _curve([(r["t"], r["elixir"], max(0.0, r["elixir"] - COST.get(r["card"], 0))
                             if r["ok"] and r["elixir"] is not None else r["elixir"]) for r in rows], t0, end, regen=False)
    live = [(r["t"], r["opp_elixir"], r["opp_elixir"]) for r in rows if r.get("opp_elixir") is not None]
    anchors = sorted([(e["t"], None, e.get("elixir_after")) for e in his] + live, key=lambda a: a[0])
    him, him_before = _curve(anchors, t0, end) if anchors else (None, [])
    him_at = dict(zip([a[0] for a in anchors], him_before))
    ev = [[r["t"] - tf, 0, r["card"], _kind(r["reason"]), b, him and _at(him, r["t"])] for r, b in zip(rows, us_before)]
    ev += [[e["t"] - tf, 1, e["card"], "", _at(us, e["t"]), him_at.get(e["t"])] for e in his]
    rd = lambda pts: [[round(x - tf, 2), round(y, 2)] for x, y in pts]  # noqa: E731
    g["chart"] = {"start": round(min(0.0, t0 - tf), 2), "end": round(end - tf, 2), "x2": round(t0 + DOUBLE_AFTER - tf, 2),
                  "us": rd(us), "him": rd(him) if him else None,
                  "ev": [[round(x, 2), w, c, k, u if u is None else round(u, 1), h if h is None else round(h, 1)]
                         for x, w, c, k, u, h in sorted(ev, key=lambda e: e[0])]}
    g["tf"], g["him_at_us"] = tf, [e[5] for e in ev[:len(rows)]]
    return g


def metrics(g: dict) -> dict:
    """Chiffres d'un match comparés entre victoires et défaites (None = inconnu pour ce match)."""
    ok = [r for r in g["rows"] if r["ok"]]
    kinds = [_kind(r["reason"]) for r in ok]
    el = [r["elixir"] for r in ok if r["elixir"] is not None]
    return {"Coups joués": len(ok) if g["rows"] else None,
            "Élixir au moment de jouer": st.mean(el) if el else None,
            "Part des défenses (%)": 100 * kinds.count("défense") / len(ok) if ok else None,
            "Géants posés": sum(r["card"] == "giant" for r in ok) if ok else None,
            "Sorts lancés": kinds.count("sort") if ok else None,
            "Élixir dépensé : nous": sum(COST.get(r["card"], 0) for r in ok) if ok else None,
            "Élixir dépensé : lui (vu)": sum(COST.get(e["card"], 0) for e in g["his"]) if g["his"] else None,
            "Bilan des échanges": g["trade"]}


def _res(r: str) -> str:
    icon, label = RESULT.get(r, ("?", "inconnu"))
    return f'<span class="res {esc(r)}"><i aria-hidden="true">{icon}</i>{label}</span>'


def _chips(items) -> str:
    return "".join(f'<span class="chip">{esc(str(c))}</span>' for c in items)


def _value(v) -> str:
    return {True: "oui", False: "non"}.get(v, str(v)) if isinstance(v, bool) else str(v)


def section_overview(games: list[dict], shown: set) -> tuple[str, dict]:
    done = [g for g in sorted(games, key=lambda g: g["id"]) if g["result"] in RESULT]
    n, c = len(done), collections.Counter(g["result"] for g in done)
    rate = 100 * c["win"] / n if n else 0.0
    w = min(ROLL, n)
    wins = [g["result"] == "win" for g in done]
    roll = [[i, round(100 * sum(wins[i - w + 1:i + 1]) / w, 1)] for i in range(w - 1, n)] if w else []
    last = roll[-1][1] if roll else rate
    trades = [g["trade"] for g in done if g["trade"] is not None]
    unknown = len(games) - n
    tiles = [("Parties", str(n), f"V {c['win']} · D {c['loss']} · N {c['draw']}"
              + (f" · {unknown} sans résultat" if unknown else "")),
             ("Taux de victoire", _pct(rate), "victoires / parties terminées"),
             (f"{w} dernières", _pct(last), f"{_signed(last - rate)} pts vs global" if n else "")]
    if trades:
        tiles.append(("Bilan des échanges", _signed(st.median(trades)), f"élixir gagné par match (médiane, {len(trades)} matchs)"))
    else:
        played = [g["played"] for g in done]
        tiles.append(("Coups par match", f"{st.median(played):.0f}" if played else "–", "médiane"))
    out = ['<section id="overview"><h2>Vue d\'ensemble</h2><div class="tiles">']
    out += [f'<div class="card tile"><div class="label">{a}</div><div class="value">{b}</div><div class="note">{esc(cn)}</div></div>'
            for a, b, cn in tiles]
    out.append("</div>")
    win = {"window": w, "rate": round(rate, 1), "roll": roll,
           "games": [{"d": g["date"].split()[0], "t": g["date"], "r": g["result"], "id": g["id"]} for g in done]}
    if n:
        out.append(f'<figure class="card"><h3>Taux de victoire glissant ({w} derniers matchs)</h3>'
                   f'<p class="sub">Ligne : victoires sur les {w} derniers matchs · repère : taux global · '
                   f'bandeau : ▲ victoire au-dessus, ▼ défaite en dessous.</p><div class="chart" id="winchart"></div></figure>')
    # victoires vs défaites : ce qui change en moyenne
    groups = {r: [metrics(g) for g in done if g["result"] == r] for r in ("win", "loss")}
    rows = []
    for k in metrics({"rows": [], "his": [], "trade": None}):
        mw, ml = ([m[k] for m in groups[r] if m[k] is not None] for r in ("win", "loss"))
        if mw or ml:
            a, b = (st.mean(x) if x else None for x in (mw, ml))
            diff = _signed(a - b) if a is not None and b is not None else ""
            rows.append(f'<tr><td>{esc(k)}</td><td class="num">{"–" if a is None else f"{a:.1f}"}</td>'
                        f'<td class="num">{"–" if b is None else f"{b:.1f}"}</td><td class="num">{diff}</td></tr>')
    out.append('<div class="cols"><div class="card"><h3>Victoires vs défaites (moyenne par match)</h3>'
               f'<table><thead><tr><th>Mesure</th><th class="num">victoires ({c["win"]})</th>'
               f'<th class="num">défaites ({c["loss"]})</th><th class="num">écart</th></tr></thead><tbody>'
               + "".join(rows) + "</tbody></table></div>")
    # cartes adverses : taux de victoire quand elles sont en face
    faced = collections.defaultdict(lambda: [0, 0])
    for g in done:
        for card in set(g["deck"]):
            faced[card][0] += g["result"] == "win"
            faced[card][1] += 1
    hard = sorted(((100 * v / t, t, v, card) for card, (v, t) in faced.items() if t >= 3), key=lambda r: (r[0], -r[1]))[:10]
    out.append('<div class="card"><h3>Cartes adverses les plus dures (au moins 3 matchs)</h3>')
    if hard:
        out.append('<table><thead><tr><th>Carte</th><th class="num">matchs</th><th class="num">V–D</th>'
                   '<th class="num">victoires</th><th class="meter-col"><span class="sr">barre</span></th></tr></thead><tbody>')
        out += [f'<tr><td>{esc(card)}</td><td class="num">{t}</td><td class="num">{v}–{t - v}</td>'
                f'<td class="num">{_pct(r)}</td><td class="meter-col"><div class="meter" title="{_pct(r)}">'
                f'<span style="width:{r:.0f}%"></span><i style="left:{rate:.0f}%"></i></div></td></tr>'
                for r, t, v, card in hard]
        out.append(f'</tbody></table><p class="sub">Trait vertical : taux global ({_pct(rate)}).</p>')
    else:
        out.append('<p class="sub">Pas encore assez de decks adverses connus.</p>')
    out.append("</div></div>")
    # une ligne par match, le plus récent en premier ; colonne « bilan » seulement si un match l'a enregistré
    bilan = any(g["trade"] is not None for g in games)
    out.append('<div class="card"><h3>Matchs</h3><div class="scroll"><table class="games"><thead><tr><th>Date</th>'
               '<th>Résultat</th><th class="num">Coups</th>' + ('<th class="num">Bilan</th>' if bilan else "")
               + '<th>Deck adverse</th></tr></thead><tbody>')
    for g in sorted(games, key=lambda g: g["id"], reverse=True):
        extra = f'<br><span class="muted">+{g["refused"]} refusés</span>' if g["refused"] else ""
        tag = "" if g["counted"] else ' <span class="tag">non compté</span>'
        date = f'<a href="#g-{esc(g["id"])}">{esc(g["date"])}</a>' if g["id"] in shown else esc(g["date"])
        out.append(f'<tr><td>{date}</td><td>{_res(g["result"])}{tag}</td><td class="num">{g["played"]}{extra}</td>'
                   + (f'<td class="num">{_signed(g["trade"])}</td>' if bilan else "")
                   + f'<td>{_chips(g["deck"]) or "<span class=muted>?</span>"}</td></tr>')
    out.append("</tbody></table></div></div></section>")
    return "\n".join(out), win


def section_game(g: dict, open_: bool) -> str:
    rows, ch = g["rows"], g["chart"]
    head = [f'<span class="gdate">{esc(g["date"])}</span>', _res(g["result"]), f'<span>{g["played"]} coups</span>']
    if g["trade"] is not None:
        head.append(f'<span>bilan {_signed(g["trade"])}</span>')
    if not g["counted"]:
        head.append('<span class="tag">non compté</span>')
    head.append(f'<span class="muted gid">{esc(g["id"])}</span>')
    out = [f'<details class="game" id="g-{esc(g["id"])}"{" open" if open_ else ""}><summary>{"".join(head)}</summary>'
           '<div class="body">']
    diff = {k: v for k, v in g["params"].items() if k in ST.DEFAULT and ST.DEFAULT[k] != v}   # réglages joués, pas les absents
    out.append(f'<p class="meta"><b>Deck adverse</b> {_chips(g["deck"]) or "<span class=muted>inconnu</span>"}</p>')
    if g["params"]:
        out.append('<p class="meta"><b>Réglages hors défaut</b> '
                   + (_chips(f"{k} = {_value(v)}" for k, v in diff.items()) or '<span class="muted">aucun</span>') + "</p>")
    if ch:
        keys = '<span><i class="key s1"></i>nous (lu à l\'écran)</span>'
        if ch["him"]:
            keys += '<span><i class="key s2"></i>lui (estimé : cartes vues + recharge)</span>'
        out.append(f'<figure class="plotbox"><h3>Élixir</h3><div class="legend">{keys}'
                   '<span><i class="key dotk"></i>carte posée</span></div>'
                   f'<div class="chart" data-game="{esc(g["id"])}"></div></figure>')
    if rows:
        kinds = collections.Counter(_kind(r["reason"]) for r in rows if r["ok"])
        out.append('<p class="meta"><b>Décisions</b> ' + _chips(f"{k} {n}" for k, n in kinds.most_common()) + "</p>")
        trade = any(r.get("trade") is not None for r in rows)   # colonne « échange » : matchs récents seulement
        out.append('<div class="scroll"><table class="decisions"><thead><tr><th>Temps</th><th>Décision</th>'
                   '<th class="num wrap">Élixir nous / lui</th>' + ('<th class="num">Échange</th>' if trade else "")
                   + "</tr></thead><tbody>")
        for r, h in zip(rows, g["him_at_us"]):
            ok, h = r["ok"], r.get("opp_elixir", h)
            el = ("–" if r["elixir"] is None else f'{r["elixir"]:.1f}') + ("" if h is None else f" / {h:.1f}")
            out.append(f'<tr{"" if ok else " class=refused"}><td class="num">{_mmss(r["t"] - g["tf"])}</td>'
                       f'<td><b>{esc(r["card"])}</b>{"" if ok else " <span class=tag>refusé</span>"} '
                       f'<span class="why">{esc(r["reason"])}</span></td><td class="num">{el}</td>'
                       + (f'<td class="num">{_signed(r.get("trade"))}</td>' if trade else "") + "</tr>")
        out.append("</tbody></table></div>")
    else:
        out.append('<p class="sub">Pas de journal de coups pour ce match.</p>')
    if g["his"]:
        tf = g.get("tf", g["his"][0]["t"])
        out.append('<p class="meta opp"><b>Cartes adverses</b> '
                   + " · ".join(f'{_mmss(e["t"] - tf)} {esc(e["card"])}' for e in g["his"]) + "</p>")
    out.append("</div></details>")
    return "".join(out)


def section_bandit(stats: dict | None) -> str:
    out = ['<section id="bandit"><h2>Ce que le bandit a appris</h2>']
    if not stats:
        return out[0] + '<p class="sub">Pas de statistiques (strategy_stats.json introuvable).</p></section>'
    table = ST.param_table(stats)
    wins, losses = sum(v["wins"] for v in stats.values()), sum(v["losses"] for v in stats.values())
    p0 = 100 * (wins + 1) / (wins + losses + 2)
    out.append(f'<p class="sub">Par paramètre, toutes les parties comptées jouées avec chaque valeur ({wins} V, {losses} D). '
               f'Proba de victoire a posteriori : valeur jamais jouée = a priori ({_pct(p0)}, trait vertical). '
               'Meilleure valeur en premier, en couleur.</p><div class="params">')
    for k, vals in table.items():
        out.append(f'<div class="card param"><h3>{esc(k)}</h3><table><thead><tr><th>Valeur</th><th class="num">V</th>'
                   '<th class="num">D</th><th class="num">proba</th><th class="meter-col"><span class="sr">barre</span></th>'
                   '</tr></thead><tbody>')
        for i, (v, w, l, post) in enumerate(vals):
            tags = ("" if i else ' <span class="tag">meilleure</span>') + \
                   (' <span class="tag">défaut</span>' if ST.DEFAULT.get(k) == v and type(ST.DEFAULT.get(k)) is type(v) else "") + \
                   ("" if v in ST.SPACE[k] else ' <span class="tag">ancienne</span>')
            out.append(f'<tr class="{"best" if not i else ""}{" never" if not w + l else ""}"><td>{esc(_value(v))}{tags}</td>'
                       f'<td class="num">{w}</td><td class="num">{l}</td><td class="num">{_pct(100 * post)}</td>'
                       f'<td class="meter-col"><div class="meter"><span style="width:{100 * post:.0f}%"></span>'
                       f'<i style="left:{p0:.0f}%"></i></div></td></tr>')
        out.append("</tbody></table></div>")
    out.append("</div></section>")
    return "\n".join(out)


def build(games_dir: Path, journal: Path | None, stats_path: Path | None, only: str | None = None) -> str:
    j = {r["game"]: r for r in _jsonl(journal) if "game" in r} if journal else {}
    folders = {p.name: p for p in games_dir.iterdir() if p.is_dir() and (p / "decisions.jsonl").exists()} \
        if games_dir.is_dir() else {}
    games = [load_game(gid, folders.get(gid), j.get(gid, {})) for gid in sorted(set(folders) | set(j))]
    if only and only not in folders and only not in j:
        raise SystemExit(f"match {only} introuvable dans {games_dir}")
    stats = None
    if stats_path and stats_path.exists():
        stats = ST.normalize(json.loads(stats_path.read_text(encoding="utf-8")))
    shown = [g for g in sorted(games, key=lambda g: g["id"], reverse=True) if not only or g["id"] == only]
    overview, win = section_overview(games, {g["id"] for g in shown})
    data = {"win": win, "games": {g["id"]: g["chart"] for g in shown if g["chart"]}}
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    span = f"du {games[0]['date'].split()[0]} au {games[-1]['date'].split()[0]}" if games else "aucun match"
    src = " · ".join(f"{a} : {esc(str(b))}" for a, b in (("matchs", games_dir), ("journal", journal), ("stats", stats_path)))
    body = [f'<header><h1>Bilan des matchs</h1><p class="sub">{len(games)} matchs · {span} · '
            f'généré le {time.strftime("%d/%m/%Y %H:%M")}</p>'
            '<nav><a href="#overview">Vue d\'ensemble</a><a href="#games">Matchs</a>'
            '<a href="#bandit">Stratégie apprise</a></nav></header>',
            overview,
            f'<section id="games"><h2>{"Match " + esc(only) if only else "Matchs en détail"}</h2>'
            '<p class="sub">Cliquer pour déplier. Survol ou flèches du clavier sur le graphique : valeurs à cet instant.</p>'
            + "".join(section_game(g, open_=bool(only) or i == 0) for i, g in enumerate(shown)) + "</section>",
            section_bandit(stats),
            f'<footer class="sub">Sources — {src}</footer>']
    return (f'<!doctype html>\n<html lang="fr"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1"><title>Bilan des matchs</title>'
            f'<style>{CSS}</style></head><body><main>{"".join(body)}</main><div id="tip" role="status" hidden></div>'
            f'<script type="application/json" id="report-data">{blob}</script><script>{JS}</script></body></html>\n')


CSS = """
:root{color-scheme:light;--plane:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;
--axis:#c3c2b7;--border:rgba(11,11,11,.10);--wash:rgba(11,11,11,.05);--s1:#2a78d6;--s2:#eb6834;--good:#0ca30c;
--bad:#d03b3b;--draw:#898781;--rest:#b5b4ad;--shadow:rgba(0,0,0,.12)}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--plane:#0d0d0d;--surface:#1a1a19;
--ink:#fff;--ink2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--wash:rgba(255,255,255,.06);
--s1:#3987e5;--s2:#d95926;--rest:#6b6a65;--shadow:rgba(0,0,0,.5)}}
:root[data-theme="dark"]{color-scheme:dark;--plane:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--grid:#2c2c2a;
--axis:#383835;--border:rgba(255,255,255,.10);--wash:rgba(255,255,255,.06);--s1:#3987e5;--s2:#d95926;--rest:#6b6a65;
--shadow:rgba(0,0,0,.5)}
*{box-sizing:border-box}
body{margin:0;background:var(--plane);color:var(--ink);font:15px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1080px;margin:0 auto;padding:20px 16px 48px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:32px 0 10px}
h3{font-size:14px;margin:0 0 8px;color:var(--ink2);font-weight:600}
a{color:inherit;text-underline-offset:2px}
nav{margin-top:8px;display:flex;flex-wrap:wrap;gap:4px 16px;font-size:14px}nav a{color:var(--ink2)}
.sub{color:var(--ink2);font-size:13px;margin:0 0 8px}.muted{color:var(--muted)}
.sr{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0)}
.card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:14px 16px;margin:0 0 12px;min-width:0}
figure{margin:0}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:0 0 12px}
.tile{margin:0}.tile .label{font-size:13px;color:var(--ink2)}.tile .value{font-size:30px;font-weight:600;line-height:1.25}
.tile .note{font-size:12px;color:var(--ink2)}
.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,340px),1fr));gap:0 12px}
table{border-collapse:collapse;width:100%;font-size:13px}
th{text-align:left;font-weight:600;color:var(--ink2);border-bottom:1px solid var(--axis);padding:5px 8px 5px 0;vertical-align:bottom}
td{border-bottom:1px solid var(--grid);padding:5px 8px 5px 0;vertical-align:top}
tr:last-child td{border-bottom:0}
.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.scroll{overflow-x:auto}
.res{white-space:nowrap}.res i{font-style:normal;font-size:11px;margin-right:4px}
.res.win i{color:var(--good)}.res.loss i{color:var(--bad)}.res.draw i{color:var(--draw)}
.chip{display:inline-block;background:var(--wash);border-radius:999px;padding:0 7px;margin:1px 3px 1px 0;font-size:12px;white-space:nowrap}
.tag{display:inline-block;font-size:11px;font-weight:400;color:var(--ink2);border:1px solid var(--border);border-radius:4px;padding:0 4px;margin-left:4px;white-space:nowrap}
details.game{background:var(--surface);border:1px solid var(--border);border-radius:10px;margin:0 0 8px}
details.game>summary{cursor:pointer;padding:10px 14px;display:flex;flex-wrap:wrap;gap:2px 14px;align-items:baseline;list-style:none}
details.game>summary::-webkit-details-marker{display:none}
details.game>summary::before{content:"▸";color:var(--muted);width:10px}
details.game[open]>summary::before{content:"▾"}
details.game .gdate{font-weight:600}details.game .gid{font-size:12px;margin-left:auto}
details.game .body{padding:0 14px 14px}
.meta{font-size:13px;margin:0 0 8px}.meta b{font-weight:600;color:var(--ink2);margin-right:6px}
.opp{color:var(--ink2);margin-top:10px}
.plotbox{margin:8px 0 12px}
.legend{display:flex;flex-wrap:wrap;gap:2px 16px;font-size:12px;color:var(--ink2);margin:0 0 4px}
.key{display:inline-block;width:14px;height:0;border-top:2px solid;vertical-align:middle;margin-right:6px}
.key.s1{border-color:var(--s1)}.key.s2{border-color:var(--s2)}
.key.dotk{width:8px;height:8px;border:0;border-radius:50%;background:var(--muted)}
.chart{width:100%;min-height:200px;touch-action:pan-y}
svg.plot{display:block;overflow:visible;outline:none}
svg.plot:focus-visible{outline:2px solid var(--s1);outline-offset:2px;border-radius:4px}
.plot .grid{stroke:var(--grid)}.plot .axis{stroke:var(--axis)}.plot .ref{stroke:var(--axis)}
.plot .tick{fill:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}
.plot .reflab,.plot .endlab{fill:var(--ink2);font-size:11px}
.plot .line{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.plot .line.s1{stroke:var(--s1)}.plot .line.s2{stroke:var(--s2)}
.plot .dot{stroke:var(--surface);stroke-width:2}.plot .dot.s1{fill:var(--s1)}.plot .dot.s2{fill:var(--s2)}
.plot .dot.hl{stroke-width:2.5}
.plot .cross{stroke:var(--muted)}
.plot .win{fill:var(--good)}.plot .loss{fill:var(--bad)}.plot .draw{fill:var(--draw)}
#tip{position:fixed;left:0;top:0;pointer-events:none;background:var(--surface);color:var(--ink);border:1px solid var(--border);
border-radius:8px;padding:6px 9px;font-size:12px;box-shadow:0 4px 14px var(--shadow);max-width:280px;z-index:10}
#tip[hidden]{display:none}.tip-title{color:var(--ink2);margin-bottom:2px}
.tip-row{display:flex;gap:6px;align-items:baseline}.tip-row b{font-weight:600;font-variant-numeric:tabular-nums}
.tip-row .key{width:12px;margin:0}
.params{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,300px),1fr));gap:0 12px}
.meter-col{width:32%;min-width:70px}
.meter{position:relative;height:8px;background:var(--wash);border-radius:4px;margin-top:5px}
.meter span{position:absolute;left:0;top:0;bottom:0;background:var(--rest);border-radius:0 4px 4px 0}
.best .meter span,.cols .meter span{background:var(--s1)}
.meter i{position:absolute;top:-3px;bottom:-3px;width:1px;background:var(--ink2)}
.best td:first-child{font-weight:600}.never td{color:var(--muted)}
.decisions td:nth-child(2) b{font-weight:600;margin-right:4px}.decisions .why{color:var(--ink2)}
th.wrap{white-space:normal}
.refused td{color:var(--muted)}
footer{margin-top:32px;word-break:break-all}
@media (max-width:560px){.tile .value{font-size:26px}.games{font-size:12px}.games td:first-child{white-space:normal}details.game .gid{display:none}.decisions{font-size:12px}
details.game>summary{padding:10px}details.game .body{padding:0 10px 12px}.card{padding:12px}}
"""

JS = r"""
(() => {
const D = JSON.parse(document.getElementById('report-data').textContent);
const probe = document.createElement('div'); probe.innerHTML = '<svg/>';
const NS = probe.firstChild.namespaceURI;   // espace de noms SVG (sans URL écrite en dur)
const tip = document.getElementById('tip');
const pct = v => Math.round(v) + '\u202f%';
const mmss = s => (s < 0 ? '-' : '') + Math.floor(Math.abs(s) / 60) + ':' + String(Math.floor(Math.abs(s) % 60)).padStart(2, '0');
function S(tag, attrs, parent, text) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (text != null) e.textContent = text;
  if (parent) parent.appendChild(e);
  return e;
}
function fillTip(s) {
  tip.textContent = '';
  const h = document.createElement('div'); h.className = 'tip-title'; h.textContent = s.title; tip.appendChild(h);
  for (const [cls, val, lab] of s.rows) {
    const row = document.createElement('div'); row.className = 'tip-row';
    const key = document.createElement('i'); key.className = 'key ' + cls;
    const b = document.createElement('b'); b.textContent = val;
    const l = document.createElement('span'); l.textContent = lab;
    row.append(key, b, l); tip.appendChild(row);
  }
}
function place(x, y) {
  const r = tip.getBoundingClientRect();
  let left = x + 14, top = y + 14;
  if (left + r.width > innerWidth - 8) left = Math.max(8, x - r.width - 14);
  if (top + r.height > innerHeight - 8) top = Math.max(8, y - r.height - 14);
  tip.style.left = left + 'px'; tip.style.top = top + 'px';
}
// Graphique en lignes : axes discrets, repères, points, bandeau V/D, réticule + infobulle (souris, toucher, clavier)
function chart(box, c) {
  box.textContent = '';
  const W = Math.max(box.clientWidth, 240), H = c.h;
  const m = {l: 34, r: c.endLabels ? 36 : 14, t: 12, b: 24 + (c.strip ? 36 : 0)};
  const pw = W - m.l - m.r, ph = H - m.t - m.b;
  const svg = S('svg', {width: W, height: H, class: 'plot', tabindex: 0, role: 'img', 'aria-label': c.label}, box);
  const sx = v => m.l + (v - c.x[0]) / ((c.x[1] - c.x[0]) || 1) * pw;
  const sy = v => m.t + ph - (v - c.y[0]) / (c.y[1] - c.y[0]) * ph;
  for (const [v, t] of c.yt) {
    S('line', {x1: m.l, x2: m.l + pw, y1: sy(v), y2: sy(v), class: v === c.y[0] ? 'axis' : 'grid'}, svg);
    S('text', {x: m.l - 6, y: sy(v) + 4, 'text-anchor': 'end', class: 'tick'}, svg, t);
  }
  const xl = m.t + ph + (c.strip ? 36 : 0) + 16;
  for (const [v, t] of c.xt) S('text', {x: sx(v), y: xl, 'text-anchor': 'middle', class: 'tick'}, svg, t);
  for (const v of c.vl || []) if (v.x > c.x[0] && v.x < c.x[1]) {
    S('line', {x1: sx(v.x), x2: sx(v.x), y1: m.t, y2: m.t + ph, class: 'ref'}, svg);
    S('text', {x: sx(v.x) + 4, y: m.t + 9, class: 'reflab'}, svg, v.label);
  }
  for (const r of c.refs || []) {
    S('line', {x1: m.l, x2: m.l + pw, y1: sy(r.y), y2: sy(r.y), class: 'ref'}, svg);
    S('text', {x: m.l + 4, y: sy(r.y) - 5, class: 'reflab'}, svg, r.label);
  }
  if (c.strip) {   // une marque par match : victoire vers le haut, défaite vers le bas (position + couleur)
    const yc = m.t + ph + 20, bw = Math.max(2, Math.min(8, pw / c.strip.length - 2));
    S('line', {x1: m.l, x2: m.l + pw, y1: yc, y2: yc, class: 'grid'}, svg);
    S('text', {x: m.l - 6, y: yc - 3, 'text-anchor': 'end', class: 'tick'}, svg, 'V');
    S('text', {x: m.l - 6, y: yc + 11, 'text-anchor': 'end', class: 'tick'}, svg, 'D');
    for (const [x, r] of c.strip) {
      const y = r === 'win' ? yc - 10 : r === 'loss' ? yc + 1 : yc - 1;
      S('rect', {x: sx(x) - bw / 2, y, width: bw, height: r === 'draw' ? 2 : 9, rx: 1, class: r}, svg);
    }
  }
  for (const s of c.series) if (s.pts.length)
    S('path', {d: 'M' + s.pts.map(p => sx(p[0]).toFixed(1) + ',' + sy(p[1]).toFixed(1)).join('L'), class: 'line ' + s.cls}, svg);
  if (c.endLabels) {   // noms en bout de ligne seulement s'ils ne se chevauchent pas (sinon la légende suffit)
    const ends = c.series.filter(s => s.pts.length).map(s => [s, sy(s.pts[s.pts.length - 1][1])]);
    if (ends.length < 2 || Math.abs(ends[0][1] - ends[1][1]) >= 14)
      for (const [s, y] of ends) S('text', {x: m.l + pw + 6, y: y + 4, class: 'endlab'}, svg, s.name);
  }
  for (const d of c.dots || []) if (d[1] != null) S('circle', {cx: sx(d[0]), cy: sy(d[1]), r: pw < 500 ? 3 : 4, class: 'dot ' + d[2]}, svg);
  const snaps = c.snaps || [];
  if (!snaps.length) return;
  const cross = S('line', {y1: m.t, y2: m.t + ph, class: 'cross', visibility: 'hidden'}, svg);
  const marks = S('g', {}, svg);
  const xs = snaps.map(s => sx(s.x));
  let cur = -1;
  function pick(i, cx, cy) {
    cur = i; const s = snaps[i];
    cross.setAttribute('x1', xs[i]); cross.setAttribute('x2', xs[i]); cross.setAttribute('visibility', 'visible');
    marks.textContent = '';
    for (const [cls, y] of s.ys) if (y != null) S('circle', {cx: xs[i], cy: sy(y), r: 5, class: 'dot hl ' + cls}, marks);
    fillTip(s); tip.hidden = false;
    if (cx == null) { const r = svg.getBoundingClientRect(); cx = r.left + xs[i]; cy = r.top + m.t; }
    place(cx, cy);
  }
  function hide() { cross.setAttribute('visibility', 'hidden'); marks.textContent = ''; tip.hidden = true; }
  function nearest(px) { let b = 0; for (let i = 1; i < xs.length; i++) if (Math.abs(xs[i] - px) < Math.abs(xs[b] - px)) b = i; return b; }
  const move = e => { const r = svg.getBoundingClientRect(); pick(nearest(e.clientX - r.left), e.clientX, e.clientY); };
  svg.addEventListener('pointermove', move);
  svg.addEventListener('pointerdown', move);
  svg.addEventListener('pointerleave', hide);
  svg.addEventListener('blur', hide);
  svg.addEventListener('focus', () => pick(cur < 0 ? snaps.length - 1 : cur));
  svg.addEventListener('keydown', e => {
    if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
      e.preventDefault(); pick(Math.max(0, Math.min(snaps.length - 1, cur + (e.key === 'ArrowRight' ? 1 : -1))));
    } else if (e.key === 'Escape') hide();
  });
}
function winChart(box) {
  const w = D.win, n = w.games.length, W = box.clientWidth;
  const every = Math.max(1, Math.ceil(n / Math.max(2, Math.floor((W - 60) / 64))));
  const xt = []; let day = '';   // date au premier repère de chaque jour, heure sinon
  for (let i = 0; i < n; i += every) { const g = w.games[i]; xt.push([i, g.d === day ? g.t.split(' ')[1] : g.d]); day = g.d; }
  const roll = new Map(w.roll.map(([i, v]) => [i, v]));
  chart(box, {h: 240, x: [-0.5, n - 0.5], y: [0, 100], yt: [0, 25, 50, 75, 100].map(v => [v, v + '%']), xt,
    series: [{cls: 's1', name: 'glissant', pts: w.roll}], refs: [{y: w.rate, label: 'global ' + pct(w.rate)}],
    strip: w.games.map((g, i) => [i, g.r]), label: 'Taux de victoire glissant',
    snaps: w.games.map((g, i) => ({x: i, title: g.t + ' · ' + {win: 'victoire', loss: 'défaite', draw: 'égalité'}[g.r],
      ys: [['s1', roll.has(i) ? roll.get(i) : null]],
      rows: [['s1', roll.has(i) ? pct(roll.get(i)) : '–', roll.has(i) ? 'sur les ' + w.window + ' derniers' : 'moins de ' + w.window + ' matchs']]}))});
}
function gameChart(box) {
  const g = D.games[box.dataset.game]; if (!g) return;
  const span = g.end - g.start, W = box.clientWidth;
  const step = [10, 15, 30, 60, 120].find(s => span / s * 52 <= W - 70) || 120;
  const xt = []; for (let v = Math.ceil(g.start / step) * step; v <= g.end; v += step) xt.push([v, mmss(v)]);
  const series = [{cls: 's1', name: 'nous', pts: g.us}];
  if (g.him) series.push({cls: 's2', name: 'lui', pts: g.him});
  chart(box, {h: 220, x: [g.start, g.end], y: [0, 10], yt: [0, 2, 4, 6, 8, 10].map(v => [v, String(v)]), xt, series,
    endLabels: !!g.him, vl: [{x: g.x2, label: 'élixir x2'}], label: 'Élixir des deux camps',
    dots: g.ev.map(e => e[1] === 0 ? [e[0], e[4], 's1'] : [e[0], e[5], 's2']),
    snaps: g.ev.map(([x, who, card, kind, u, h]) => ({x, title: mmss(x) + (who ? ' · il pose ' : ' · on pose ') + card,
      ys: [['s1', u], ['s2', h]],
      rows: [['s1', u == null ? '–' : u.toFixed(1), 'nous' + (who === 0 ? ' · ' + card + (kind ? ' (' + kind + ')' : '') : '')]]
        .concat(g.him ? [['s2', h == null ? '–' : h.toFixed(1), 'lui' + (who === 1 ? ' · ' + card : '')]] : [])}))});
}
const draw = box => (box.dataset.game ? gameChart : winChart)(box);
const visible = () => document.querySelectorAll('.chart').forEach(b => { if (b.offsetParent !== null) draw(b); });
document.querySelectorAll('details.game').forEach(d => d.addEventListener('toggle', () => {
  if (d.open) d.querySelectorAll('.chart').forEach(draw);
}));
let timer, lastW = innerWidth;
addEventListener('resize', () => {
  if (innerWidth === lastW) return;
  lastW = innerWidth; clearTimeout(timer); timer = setTimeout(visible, 150);
});
// lien vers un match (#g-...) : le déplier
const openHash = () => {
  const d = document.getElementById(decodeURIComponent(location.hash.slice(1)));
  if (d && d.tagName === 'DETAILS') d.open = true;
};
addEventListener('hashchange', openHash);
openHash();
visible();
})();
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--games", default=str(ROOT / "runs/games"))
    ap.add_argument("--journal", help="défaut : <games>/journal.jsonl, sinon learning/journal.jsonl")
    ap.add_argument("--stats", help="défaut : runs/strategy_stats.json, sinon learning/strategy_stats.json")
    ap.add_argument("--out", default=str(ROOT / "runs/report.html"))
    ap.add_argument("--game", help="un seul match en détail (identifiant, ex. 20260927-083411)")
    a = ap.parse_args(argv)
    games = Path(a.games)
    first = lambda *ps: next((Path(p) for p in ps if Path(p).exists()), None)  # noqa: E731
    journal = Path(a.journal) if a.journal else first(games / "journal.jsonl", ROOT / "learning/journal.jsonl")
    stats = Path(a.stats) if a.stats else first(ROOT / "runs/strategy_stats.json", ROOT / "learning/strategy_stats.json")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(games, journal, stats, a.game), encoding="utf-8")
    print(out.resolve())
    return out


if __name__ == "__main__":
    main()
