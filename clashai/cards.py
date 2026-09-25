"""Ce que l'IA sait de chaque carte : coût, type, ce qu'elle cible, son rôle.

Pour l'instant le deck du compte « claude » (deck de départ). Les noms des unités
correspondent aux classes du détecteur (label_list de KataCR).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Card:
    name: str
    cost: int
    kind: str                  # "troop" | "spell" | "building"
    targets: str = "ground"    # ce que la carte peut toucher : "ground" | "air+ground" | "buildings"
    flying: bool = False       # l'unité posée vole-t-elle ?
    role: str = ""             # "tank" | "dps" | "swarm" | "mini-tank" | "spell-small" | "spell-big"
    radius: float = 0.0        # rayon d'effet d'un sort (fraction de largeur d'arène)
    units: tuple[str, ...] = field(default_factory=tuple)   # classes du détecteur produites


DECK = {c.name: c for c in [
    Card("giant", 5, "troop", "buildings", role="tank", units=("giant",)),
    Card("knight", 3, "troop", role="mini-tank", units=("knight",)),
    Card("mini-pekka", 4, "troop", role="dps", units=("mini-pekka",)),
    Card("cannon", 3, "building", role="defense-building", units=("cannon",)),
    Card("archers", 3, "troop", "air+ground", role="dps", units=("archer",)),
    Card("valkyrie", 4, "troop", role="splash", units=("valkyrie",)),
    Card("arrows", 3, "spell", "air+ground", role="spell-small", radius=0.14),
    Card("fireball", 4, "spell", "air+ground", role="spell-big", radius=0.11),
]}

# Cartes connues hors deck (utile si on les remet)
BENCH = {c.name: c for c in [
    Card("musketeer", 4, "troop", "air+ground", role="dps", units=("musketeer",)),
    Card("minions", 3, "troop", "air+ground", flying=True, role="swarm", units=("minion",)),
    Card("spear-goblins", 2, "troop", "air+ground", role="swarm", units=("spear-goblin",)),
]}

# Unités ennemies : ce qui compte pour choisir une réponse (valeurs approximatives)
AIR_UNITS = {"minion", "bat", "mega-minion", "baby-dragon", "inferno-dragon", "balloon", "lava-hound",
             "lava-pup", "electro-dragon", "skeleton-dragon", "phoenix-big", "phoenix-small", "flying-machine",
             "bat-evolution", "minion-horde"}
SWARM_UNITS = {"skeleton", "goblin", "spear-goblin", "bat", "minion", "barbarian", "archer",
               "royal-recruit", "guard", "fire-spirit", "ice-spirit", "electro-spirit", "heal-spirit",
               "lava-pup", "bat-evolution", "skeleton-evolution", "barbarian-evolution"}
TANK_UNITS = {"giant", "golem", "pekka", "mega-knight", "royal-giant", "electro-giant", "goblin-giant",
              "lava-hound", "giant-skeleton", "elixir-golem-big", "balloon", "hog", "ram-rider", "battle-ram"}
BUILDINGS = {"king-tower", "queen-tower", "cannoneer-tower", "dagger-duchess-tower", "cannon", "tesla",
             "inferno-tower", "bomb-tower", "mortar", "x-bow", "goblin-hut", "furnace", "barbarian-hut",
             "tombstone", "elixir-collector", "goblin-cage", "goblin-drill"}

# Effets de sorts et objets au sol : visibles, mais ce ne sont pas des unités à combattre
NOT_UNITS = {"arrows", "fireball", "zap", "zap-evolution", "poison", "earthquake", "freeze", "rage",
             "lightning", "rocket", "the-log", "tornado", "giant-snowball", "barbarian-barrel",
             "royal-delivery", "graveyard", "clone", "mirror", "goblin-barrel", "skeleton-barrel",
             "dirt", "bomb", "axe", "goblin-ball", "skeleton-king-skill", "tesla-evolution-shock",
             "selected", "text", "phoenix-egg"}
