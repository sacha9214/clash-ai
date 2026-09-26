"""Associe chaque rendu du Fan Kit (titre de fichier) à une classe du détecteur.

Noms des classes comme KataCR : unité au singulier, « -evolution » pour une évolution, « -hero » pour un héros
(ex. IceWizardHero_Pose03_4K -> ice-wizard-hero ; barbarianEvo_Pose01_4K_FX -> barbarian-evolution).
Ce qui n'est pas une unité en jeu (bannières, cartes, miniatures, fragments, le Roi…) est écarté.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

META = Path("D:/clash-ai-dataset/fankit/characters.json")
SKIP = ("banner", "card", "thumbnail", "shard", "youtube", "loading", "emote", "group", "logo", "chest", "promo",
        "season", "pass", "trello", "background", "frame", "icon", "badge", "king", "tower", "sticker", "portrait",
        "shadow", "silhouette", "merge", "pin", "doctor",
        # morceaux et personnages hors échelle / hors ladder
        "sword", "axe", "helmet", "lance", "pony", "gun", "turret", "hanmmer", "hammer", "ball", "8bit",
        "goblin-queen", "goblinqueen", "baby-goblin", "babygoblin", "trader", "goblin-knight", "goblinknight", "chef",
        "totem", "little-prince-guard", "little_prince_guard", "royal-guard", "zap-barbarian", "zap_barbarian")
NOISE = {"pose", "fx", "vfx", "a", "b", "c", "eyes", "alpha", "hd", "final", "new", "v", "render", "the", "full",
         "body", "and", "with", "clash", "royale", "4k", "2k", "8k", "art",
         # variantes d'un même rendu (fond, couleur, angle, effets) : même classe
         "back", "base", "blue", "white", "green", "purple", "red", "regular", "transparent", "glowy", "glowyeyes", "glow",
         "neutral", "no", "nofx", "generic", "psd", "solo", "pancake", "pancakes", "arrows", "infoscreen", "info",
         "screen", "ground", "png", "02png", "s05", "bac", "smoke", "16x9", "9x16", "character", "christmas", "spell",
         "ev", "heroes", "golem2"}
# nom de carte -> nom d'unité du détecteur (pluriels, noms différents)
ALIAS = {"barbarians": "barbarian", "elite-barbarians": "elite-barbarian", "archers": "archer", "skeletons": "skeleton",
         "bats": "bat", "wall-breakers": "wall-breaker", "royal-recruits": "royal-recruit", "royal-hogs": "royal-hog",
         "goblins": "goblin", "spear-goblins": "spear-goblin", "minions": "minion", "minion-horde": "minion",
         "minon-horde": "minion", "skeleton-army": "skeleton", "guards": "guard", "rascals": "rascal-boy",
         "zappies": "zappy", "pekka": "pekka", "p-e-k-k-a": "pekka", "mini-p-e-k-k-a": "mini-pekka", "minipekka": "mini-pekka",
         "hog": "hog-rider", "hogrider": "hog-rider", "log": "the-log", "snowball": "giant-snowball", "skeleton-dragons":
         "skeleton-dragon", "three-musketeers": "musketeer", "goblin-gang": "goblin", "phoenix": "phoenix-big",
         "goblinstein-monster": "goblinstein", "bush-goblin": "suspicious-bush", "vine-spell": "vines", "vine": "vines",
         "berserk": "berserker", "champion-big-bandit": "boss-bandit", "big-bandit": "boss-bandit",
         "magicarcher": "magic-archer", "lighting": "lightning", "three-musk": "musketeer", "3musketeers": "musketeer",
         "recruits": "royal-recruit", "ghost": "royal-ghost", "fire-spirits": "fire-spirit", "furnace-fire-spirit": "fire-spirit",
         "ice-golem-golem": "ice-golem", "skeleton-army": "skeleton", "minion-horde": "minion", "prince-royal-guard-prince": "prince",
         "little-prince": "little-prince", "mega-minion": "mega-minion", "princess-16x9": "princess"}


def tokens(title: str) -> list[str]:
    t = re.sub(r"([a-z])([A-Z])", r"\1 \2", title)          # CamelCase -> mots
    t = re.sub(r"[_\-\.\s]+", " ", t).lower()
    return [w for w in t.split() if w not in NOISE and not re.fullmatch(r"\d+|pose\d+|\d+k|0\d|v\d+", w)]


def to_class(title: str) -> str | None:
    low = title.lower()
    if any(s in low for s in SKIP):
        return None
    w = tokens(title)
    suffix = ""
    if "hero" in w:
        suffix, w = "-hero", [x for x in w if x != "hero"]
    if "evo" in w or "evolution" in w or "evolved" in w:
        suffix, w = "-evolution", [x for x in w if x not in ("evo", "evolution", "evolved")]
    if not w:
        return None
    base = "-".join(w)
    base = ALIAS.get(base, base)
    return base + suffix


def mapping() -> dict[int, str]:
    meta = json.load(open(META, encoding="utf-8"))
    return {a["id"]: c for a in meta if (c := to_class(a["title"]))}


if __name__ == "__main__":
    import collections
    meta = json.load(open(META, encoding="utf-8"))
    m = {a["id"]: (a["title"], to_class(a["title"])) for a in meta}
    cnt = collections.Counter(c for _, c in m.values() if c)
    print(f"{sum(cnt.values())} rendus retenus sur {len(meta)}, {len(cnt)} classes")
    for c, n in sorted(cnt.items()):
        print(f"  {c:32s} {n}")
