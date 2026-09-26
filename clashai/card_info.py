"""Statistiques des cartes pour le cerveau (portée, vitesse, cibles, zone, coût…), en cases et secondes.

Source : data/cards/cards_db.json (scripts/build_card_db.py) — fichiers du jeu (RoyaleAPI) complétés et mis à jour
par le wiki officiel. Les valeurs « current » (wiki, actuelles) passent avant celles de 2023.
Les noms d'unités du détecteur (ex. « hog-rider », « archer ») sont reliés aux cartes par opponent.UNIT2CARD.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

DB_FILE = Path(__file__).resolve().parents[1] / "data/cards/cards_db.json"


@lru_cache(maxsize=1)
def db() -> dict:
    return json.loads(DB_FILE.read_text(encoding="utf-8")) if DB_FILE.exists() else {}


def _num(s) -> float | None:
    """« Melee: Medium (1.2) » -> 1.2 ; « 2.5-5 » -> 5 ; « 6 » -> 6."""
    if s is None:
        return None
    vals = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", str(s))]
    return max(vals) if vals else None


@lru_cache(maxsize=512)
def card_for_unit(unit: str) -> str | None:
    from clashai.opponent import UNIT2CARD
    if unit in UNIT2CARD:
        return UNIT2CARD[unit][0]
    return unit if unit in db() else None


@lru_cache(maxsize=512)
def info(unit: str) -> dict:
    """Portée (cases), vitesse (cases/s), cibles, zone, volant, vise seulement les bâtiments, coût."""
    card = card_for_unit(unit)
    e = db().get(card or "", {})
    u = (e.get("units") or [{}])[0]
    cur = (e.get("current") or {}).get("attributes") or e.get("attributes") or {}
    rng = _num(cur.get("Range")) if cur.get("Range") else u.get("range_tiles")
    speed = _num(cur.get("Speed"))
    speed = speed / 60 if speed else u.get("speed_tiles_s")
    target = (cur.get("Target") or u.get("targets") or "").lower()
    return {
        "card": card,
        "range": rng if rng else None,
        "speed": speed,
        "hit_speed": _num(cur.get("Hit Speed")) or u.get("hit_speed_s"),
        "buildings_only": "building" in target or u.get("targets_only") == "bâtiments",
        "hits_air": "air" in target,
        "flying": (cur.get("Transport") or "").lower() == "air" or bool(u.get("flying")),
        "splash": _num(cur.get("Radius")) or u.get("splash_radius_tiles"),
        "cost": e.get("elixir"),
    }


def attack_range(unit: str, default: float = 1.2) -> float:
    r = info(unit)["range"]
    return r if r else default


def spell_radius(card: str, default: float = 3.0) -> float:
    e = db().get(card, {})
    cur = (e.get("current") or {}).get("attributes") or e.get("attributes") or {}
    return _num(cur.get("Radius")) or (e.get("spell") or {}).get("radius_tiles") or default
