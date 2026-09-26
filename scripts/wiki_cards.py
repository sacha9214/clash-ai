"""Complète la base des cartes avec le wiki officiel (clashroyale.fandom.com) : cartes récentes, héros, évolutions.

Pour chaque page : tableau des attributs (coût, vitesse d'attaque, vitesse, déploiement, portée, cibles…),
tableau des niveaux (PV / dégâts au niveau 11 = niveau tournoi), et le texte de la capacité (héros / évolution).
Résultat : data/cards/manual.json (fusionné dans cards_db.json par build_card_db.py).

  .venv-katacr\\Scripts\\python scripts/wiki_cards.py
"""
from __future__ import annotations

import io
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
API = "https://clashroyale.fandom.com/api.php"
HEADERS = {"User-Agent": "clash-ai-personal-research/1.0 (card statistics)"}
NEW = ["Berserker", "Little Prince", "Goblinstein", "Boss Bandit", "Goblin Machine", "Ronin", "Spirit Empress",
       "Goblin Curse", "Vines", "Rune Giant", "Void", "Suspicious Bush", "Goblin Demolisher", "Minion Giant"]
HEROES = ["Knight", "Giant", "Mini P.E.K.K.A.", "Musketeer", "Wizard", "Magic Archer", "Mega Minion", "Barbarian Barrel",
          "Goblins", "Ice Golem", "Bowler", "Dark Prince", "Valkyrie", "Berserker", "Ice Wizard", "Balloon", "Tombstone"]
# évolutions : la page « Carte/Evolution »
EVOS = ["Knight", "Archers", "Skeletons", "Barbarians", "Royal Recruits", "Royal Giant", "Mortar", "Bats", "Firecracker",
        "Wall Breakers", "Ice Spirit", "Valkyrie", "Tesla", "Zap", "Bomber", "Battle Ram", "Wizard", "Royal Hogs",
        "Goblin Cage", "Goblin Barrel", "Furnace", "Skeleton Barrel", "Dart Goblin", "Goblin Giant", "Hunter",
        "Executioner", "Inferno Dragon", "Lumberjack", "Musketeer", "Electro Dragon", "Giant Snowball", "Royal Ghost",
        "Baby Dragon", "Goblin Drill", "Cannon", "Witch", "Skeleton Army", "Princess", "Elite Barbarians", "Mega Knight",
        "P.E.K.K.A"]


def page_html(title: str) -> str | None:
    r = requests.get(API, params={"action": "parse", "page": title, "prop": "text", "format": "json", "redirects": 1},
                     headers=HEADERS, timeout=30)
    j = r.json()
    return j.get("parse", {}).get("text", {}).get("*")


def page_wikitext(title: str) -> str:
    r = requests.get(API, params={"action": "parse", "page": title, "prop": "wikitext", "format": "json", "redirects": 1},
                     headers=HEADERS, timeout=30)
    return r.json().get("parse", {}).get("wikitext", {}).get("*", "")


def clean(v):
    s = re.sub(r"\[\d+\]|\s+", " ", str(v)).strip()
    return None if s in ("", "nan", "N/A") else s


def parse(title: str, kind: str) -> dict | None:
    html = page_html(title)
    if not html:
        return None
    out = {"wiki_page": title, "kind": kind}
    try:
        tables = pd.read_html(io.StringIO(html))
    except ValueError:
        tables = []
    for t in tables:
        cols = [clean(c) or "" for c in (t.columns.get_level_values(-1) if isinstance(t.columns, pd.MultiIndex) else t.columns)]
        low = " ".join(cols).lower()
        if ("range" in low or "hit speed" in low or "radius" in low) and len(t) >= 1 and "level" not in low:
            for c, v in zip(cols, t.iloc[0].tolist()):
                if c and clean(v):
                    out.setdefault("attributes", {})[c] = clean(v)
        elif low.startswith("level") and len(t) >= 11:
            row = t[t.iloc[:, 0].astype(str).str.strip() == "11"]
            if len(row):
                for c, v in zip(cols, row.iloc[0].tolist()):
                    if c and clean(v) and c.lower() != "level":
                        out.setdefault("level11", {})[c] = clean(v)
    wt = page_wikitext(title)
    m = re.search(r"\|Cost=(\d+)", wt)
    if m:
        out["elixir"] = int(m.group(1))
    m = re.search(r"\|Type=(\w+)", wt)
    if m:
        out["type"] = m.group(1)
    intro = re.sub(r"\{\{[^}]*\}\}|\[\[(?:[^|\]]*\|)?([^\]]*)\]\]|'''?|<[^>]+>", r"\1", wt.split("==")[0])
    out["description"] = re.sub(r"\s+", " ", intro).strip()[:900]
    for sec in ("Ability", "Hero Ability", "Evolution", "Evolved"):
        m = re.search(r"==+\s*" + sec + r"[^=]*==+(.*?)(?:\n==[^=]|\Z)", wt, re.S)
        if m:
            txt = re.sub(r"\{\{[^}]*\}\}|\[\[(?:[^|\]]*\|)?([^\]]*)\]\]|'''?|<[^>]+>", r"\1", m.group(1))
            out["ability"] = re.sub(r"\s+", " ", txt).strip()[:900]
            break
    return out


def key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower().replace("p.e.k.k.a", "pekka")).strip("-")


def main():
    f = ROOT / "data/cards/manual.json"
    manual = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    base = [c["name"] for c in json.loads((ROOT / "data/cards/cards.json").read_text(encoding="utf-8"))]
    jobs = [(n, n, "carte", key(n)) for n in NEW] + \
           [(n, n.replace("P.E.K.K.A", "P.E.K.K.A."), "carte (valeurs actuelles)", key(n) + "@wiki") for n in base] + \
           [(n, f"{n}/Hero", "héros", key(n) + "-hero") for n in HEROES] + \
           [(n, f"{n}/Evolution", "évolution", key(n) + "-evolution") for n in EVOS]
    ok = 0
    for name, title, kind, k in jobs:
        if k in manual and manual[k].get("attributes"):
            ok += 1
            continue
        try:
            d = parse(title, kind)
        except Exception as e:                     # une page absente ne doit pas tout arrêter
            d = None
            print(f"  erreur {title} : {e}")
        if d and (d.get("attributes") or d.get("ability")):
            d["name"] = name + ("" if kind == "carte" else f" ({kind})")
            d["source"] = f"clashroyale.fandom.com/wiki/{title.replace(' ', '_')}"
            manual[k] = d
            ok += 1
            print(f"{k:30s} {len(d.get('attributes', {}))} attributs{' + capacité' if d.get('ability') else ''}", flush=True)
        else:
            print(f"{k:30s} introuvable", flush=True)
        time.sleep(0.5)                            # rythme poli
    f.write_text(json.dumps(manual, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{ok}/{len(jobs)} pages récupérées -> {f}")


if __name__ == "__main__":
    main()
