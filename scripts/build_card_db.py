"""Base de données des cartes : portée, vitesse, vitesse d'attaque, cibles, zone, PV… en CASES et SECONDES.

Source 1 (fiable, extraite des fichiers du jeu) : RoyaleAPI/cr-api-data (docs/json, fin 2023) -> data/cards/*.json.
Source 2 (cartes plus récentes et héros/évolutions) : data/cards/manual.json, rempli depuis le wiki officiel.
Unités du jeu -> cases : 1000 = 1 case ; vitesse en cases/minute (60 = moyenne = 1 case/s) ; temps en ms.

  .venv\\Scripts\\python scripts/build_card_db.py      -> data/cards/cards_db.json
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data/cards"
SPEED_NAME = {45: "lente", 60: "moyenne", 90: "rapide", 120: "très rapide"}


def load(name):
    f = D / name
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else []


def tiles(v):
    return round(v / 1000, 2) if v else 0


PROJS: dict = {}


def unit_stats(ch: dict, count: int = 1) -> dict:
    only = ("bâtiments" if ch.get("target_only_buildings") else "tours" if ch.get("target_only_towers")
            else "troupes" if ch.get("target_only_troops") else None)
    targets = "air+sol" if ch.get("attacks_air") and ch.get("attacks_ground") else \
        "air" if ch.get("attacks_air") else "sol" if ch.get("attacks_ground") else "aucune"
    out = {
        "unit": ch["name"], "count": count,
        "hp_lvl1": ch.get("hitpoints"),
        # dégâts : ceux de l'unité, sinon ceux de son projectile (tireurs)
        "damage_lvl1": ch.get("damage") or PROJS.get(ch.get("projectile") or "", {}).get("damage") or None,
        "range_tiles": tiles(ch.get("range")), "min_range_tiles": tiles(ch.get("minimum_range")) or None,
        "sight_tiles": tiles(ch.get("sight_range")),
        "speed": ch.get("speed") or None, "speed_tiles_s": round(ch["speed"] / 60, 2) if ch.get("speed") else None,
        "speed_class": SPEED_NAME.get(ch.get("speed")),
        "hit_speed_s": round(ch["hit_speed"] / 1000, 2) if ch.get("hit_speed") else None,
        "first_hit_s": round(ch["load_time"] / 1000, 2) if ch.get("load_time") else None,
        "deploy_time_s": round(ch["deploy_time"] / 1000, 2) if ch.get("deploy_time") else None,
        "targets": targets, "targets_only": only,
        "splash_radius_tiles": tiles(ch.get("area_damage_radius")) or None,
        "melee": bool(ch.get("range")) and ch["range"] <= 1200,
        "flying": bool(ch.get("flying_height")),
        "charge_range_tiles": tiles(ch.get("charge_range")) or None,
        "death_damage_lvl1": ch.get("death_damage") or None, "death_radius_tiles": tiles(ch.get("death_damage_radius")) or None,
        "death_spawn": ch.get("death_spawn_character"), "spawns": ch.get("spawn_character"),
        "spawn_interval_s": round(ch["spawn_pause_time"] / 1000, 2) if ch.get("spawn_pause_time") else None,
        "lifetime_s": round(ch["life_time"] / 1000, 1) if ch.get("life_time") else None,
        "projectile": ch.get("projectile"),
    }
    return {k: v for k, v in out.items() if v not in (None, False)}


def main():
    cards = load("cards.json")
    chars = {c["name"]: c for c in load("cards_stats_characters.json")}
    chars.update({c["name"]: c for c in load("cards_stats_building.json")})
    troop = {c.get("key"): c for c in load("cards_stats_troop.json")}
    spells = {c.get("key"): c for c in load("cards_stats_spell.json")}
    projs = {p["name"]: p for p in load("cards_stats_projectile.json")}
    PROJS.update(projs)
    db = {}
    for c in cards:
        k = c["key"]
        e = {"name": c["name"], "elixir": c["elixir"], "type": c["type"], "rarity": c["rarity"],
             "description": c.get("description"), "has_evolution": bool(c.get("evolved_spells_sc_key"))}
        t = troop.get(k) or spells.get(k) or {}
        summon = t.get("summon_character") or c.get("sc_key")
        n = t.get("summon_number") or 1
        if summon in chars:
            e["units"] = [unit_stats(chars[summon], n)]
            second = t.get("summon_character_second")
            if second and second in chars:
                e["units"].append(unit_stats(chars[second], t.get("summon_character_second_count") or 1))
        if c["type"] == "Spell":
            sp = spells.get(k, {})
            proj = projs.get(t.get("projectile") or sp.get("projectile") or "") or {}
            if not proj:                      # sorts à projectile : FireballSpell, ArrowsSpell, RocketSpell, LogProjectile…
                sc = c.get("sc_key", "")
                proj = next((projs[n] for n in (sc + "Spell", sc.replace("The", "") + "Projectile", sc + "Projectile",
                                                 "LighningSpell" if sc == "Lightning" else "") if n in projs), {})
            e["spell"] = {k2: v for k2, v in {
                "radius_tiles": tiles(sp.get("radius") or proj.get("radius")),
                "duration_s": round(sp["life_duration"] / 1000, 1) if sp.get("life_duration") else None,
                "damage_lvl1": proj.get("damage") or sp.get("damage") or None,
                "projectile_speed": proj.get("speed") or None,
                "hits_air": sp.get("hits_air", proj.get("aoe_to_air")), "hits_ground": sp.get("hits_ground", proj.get("aoe_to_ground")),
                "tower_damage_pct": proj.get("crown_tower_damage_percent") or None,
                "pushback_tiles": tiles(proj.get("pushback")) or None,
            }.items() if v not in (None, 0)}
        e["source"] = "RoyaleAPI cr-api-data (fichiers du jeu, 2023)"
        db[k] = e
    manual = load("manual.json")
    for k, v in (manual.items() if isinstance(manual, dict) else []):
        if k.endswith("@wiki"):                   # valeurs actuelles du wiki pour une carte déjà connue
            base = k[:-5]
            if base in db:
                db[base]["current"] = {x: v.get(x) for x in ("attributes", "level11", "ability") if v.get(x)}
                db[base]["current_source"] = v.get("source")
            continue
        db[k] = {**db.get(k, {}), **v}
    (D / "cards_db.json").write_text(json.dumps(db, ensure_ascii=False, indent=1), encoding="utf-8")
    troops = [k for k, v in db.items() if v.get("type") == "Troop"]
    no_units = [k for k in troops if not db[k].get("units")]
    print(f"{len(db)} cartes ; {len(troops)} troupes dont {len(no_units)} sans statistiques d'unité : {no_units}")


if __name__ == "__main__":
    main()
