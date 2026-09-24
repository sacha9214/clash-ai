"""Modèle de l'adversaire : son élixir, son deck, sa main et ses prochaines cartes.

- Élixir : il commence à 5, gagne 1 toutes les 2,8 s (2x plus vite en fin de
  partie), plafonné à 10 ; chaque carte posée coûte son prix. On n'estime pas au
  dixième près (on ne voit pas tout), mais l'ordre de grandeur suffit pour punir
  un adversaire à sec.
- Cartes : une nouvelle unité ennemie (nouvel identifiant de suivi) = une carte
  posée ; plusieurs unités identiques qui apparaissent ensemble (3 gobelins) = une
  seule carte.
- Cycle : un deck fait 8 cartes, 4 en main ; la carte jouée passe derrière la
  file. Donc une carte jouée ne revient en main qu'après 4 autres cartes :
  main probable = deck connu moins les 4 dernières jouées ; la prochaine à
  rentrer en main = celle jouée il y a 4 cartes.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

# unité détectée -> (carte, coût, nombre d'unités par carte)
UNIT2CARD = {
    "knight": ("knight", 3, 1), "archer": ("archers", 3, 2), "goblin": ("goblins", 2, 3),
    "spear-goblin": ("spear-goblins", 2, 3), "giant": ("giant", 5, 1), "mini-pekka": ("mini-pekka", 4, 1),
    "musketeer": ("musketeer", 4, 1), "minion": ("minions", 3, 3), "skeleton": ("skeletons", 1, 3),
    "bomber": ("bomber", 2, 1), "valkyrie": ("valkyrie", 4, 1), "hog-rider": ("hog-rider", 4, 1),
    "hog": ("hog-rider", 4, 1), "barbarian": ("barbarians", 5, 5), "wizard": ("wizard", 5, 1),
    "baby-dragon": ("baby-dragon", 4, 1), "prince": ("prince", 5, 1), "witch": ("witch", 5, 1),
    "pekka": ("pekka", 7, 1), "royal-giant": ("royal-giant", 6, 1), "golem": ("golem", 8, 1),
    "mega-minion": ("mega-minion", 3, 1), "bat": ("bats", 2, 5), "ice-spirit": ("ice-spirit", 1, 1),
    "fire-spirit": ("fire-spirit", 1, 1), "electro-spirit": ("electro-spirit", 1, 1),
    "ice-golem": ("ice-golem", 2, 1), "dart-goblin": ("dart-goblin", 3, 1), "princess": ("princess", 3, 1),
    "miner": ("miner", 3, 1), "lumberjack": ("lumberjack", 4, 1), "balloon": ("balloon", 5, 1),
    "giant-skeleton": ("giant-skeleton", 6, 1), "mega-knight": ("mega-knight", 7, 1),
    "electro-wizard": ("electro-wizard", 4, 1), "ice-wizard": ("ice-wizard", 3, 1),
    "royal-hog": ("royal-hogs", 5, 4), "royal-recruit": ("royal-recruits", 7, 6), "guard": ("guards", 3, 3),
    "elite-barbarian": ("elite-barbarians", 6, 2), "dark-prince": ("dark-prince", 4, 1),
    "bandit": ("bandit", 3, 1), "firecracker": ("firecracker", 3, 1), "executioner": ("executioner", 5, 1),
    "battle-ram": ("battle-ram", 4, 1), "ram-rider": ("ram-rider", 5, 1), "inferno-dragon": ("inferno-dragon", 4, 1),
    "wall-breaker": ("wall-breakers", 2, 2), "rascal-boy": ("rascals", 5, 3), "rascal-girl": ("rascals", 5, 3),
    "cannon": ("cannon", 3, 1), "tesla": ("tesla", 4, 1), "inferno-tower": ("inferno-tower", 5, 1),
    "bomb-tower": ("bomb-tower", 4, 1), "goblin-hut": ("goblin-hut", 5, 1), "tombstone": ("tombstone", 3, 1),
    "mortar": ("mortar", 4, 1), "x-bow": ("x-bow", 6, 1), "furnace": ("furnace", 4, 1),
    "barbarian-hut": ("barbarian-hut", 7, 1), "elixir-collector": ("elixir-collector", 6, 1),
    # sorts : visibles un court instant
    "arrows": ("arrows", 3, 1), "fireball": ("fireball", 4, 1), "zap": ("zap", 2, 1), "poison": ("poison", 4, 1),
    "the-log": ("the-log", 2, 1), "rocket": ("rocket", 6, 1), "lightning": ("lightning", 6, 1),
    "freeze": ("freeze", 4, 1), "earthquake": ("earthquake", 3, 1), "tornado": ("tornado", 3, 1),
    "goblin-barrel": ("goblin-barrel", 3, 1), "barbarian-barrel": ("barbarian-barrel", 2, 1),
    "giant-snowball": ("giant-snowball", 2, 1), "graveyard": ("graveyard", 5, 1), "rage": ("rage", 2, 1),
}
# unités qui naissent d'autres unités ou bâtiments : ce ne sont pas des cartes jouées
SPELLS = {"arrows", "fireball", "zap", "poison", "the-log", "rocket", "lightning", "freeze", "earthquake",
          "tornado", "goblin-barrel", "barbarian-barrel", "giant-snowball", "graveyard", "rage"}
SPAWNED = {"golemite", "lava-pup", "elixir-golem-mid", "elixir-golem-small", "phoenix-egg", "phoenix-small"}
REGEN, MAX_ELIXIR, DOUBLE_AFTER = 1 / 2.8, 10.0, 120.0


@dataclass
class Opponent:
    start: float = field(default_factory=time.time)
    elixir: float = 5.0
    last_t: float = field(default_factory=time.time)
    played: list[str] = field(default_factory=list)       # cartes dans l'ordre joué
    seen_ids: set[int] = field(default_factory=set)
    recent: dict[str, tuple[float, int]] = field(default_factory=dict)   # carte -> (instant, unités vues)
    prev_counts: dict[str, int] = field(default_factory=dict)
    pending: dict[str, float] = field(default_factory=dict)

    def update(self, units, now: float | None = None, frame_h: int = 1280) -> list[str]:
        """Met à jour avec les unités détectées. Renvoie les cartes que l'ennemi vient de poser.

        Robuste au bruit du détecteur : une carte n'est comptée que si le NOMBRE
        d'unités ennemies de ce type augmente, qu'une nouvelle unité est apparue
        côté ennemi (on ne peut poser que chez soi), et que ça tient sur 2 images.
        Les sorts sont ignorés (effets trop brefs, confondus avec les nôtres)."""
        now = now or time.time()
        rate = REGEN * (2 if now - self.start > DOUBLE_AFTER else 1)
        self.elixir = min(MAX_ELIXIR, self.elixir + (now - self.last_t) * rate)
        self.last_t = now
        counts, fresh = {}, {}
        for u in units:
            if not u.enemy or u.name in SPAWNED or u.name not in UNIT2CARD or u.name in SPELLS:
                continue
            card = UNIT2CARD[u.name][0]
            counts[card] = counts.get(card, 0) + 1
            if u.track_id >= 0 and u.track_id not in self.seen_ids:
                self.seen_ids.add(u.track_id)
                if u.center[1] < 0.52 * frame_h:          # apparue côté ennemi
                    fresh[card] = fresh.get(card, 0) + 1
        new_cards = []
        for card, n in counts.items():
            prev = self.prev_counts.get(card, 0)
            _, cost, per = next(v for v in UNIT2CARD.values() if v[0] == card)
            if fresh.get(card) and n > prev:
                pend = self.pending.get(card)
                if pend and now - pend < 2.0:              # confirmé sur une 2e image
                    self.pending.pop(card)
                    t_last = self.recent.get(card, (-1e9, 0))[0]
                    if now - t_last > 2.5:                 # pas le même groupe qui finit d'apparaître
                        self.recent[card] = (now, n)
                        self.elixir = max(0.0, self.elixir - cost)
                        self.played.append(card)
                        new_cards.append(card)
                else:
                    self.pending[card] = now
        self.prev_counts = counts
        return new_cards

    @property
    def deck(self) -> list[str]:
        """Cartes connues de son deck, dans l'ordre de première apparition (8 max)."""
        out = []
        for c in self.played:
            if c not in out:
                out.append(c)
        return out[:8]

    @property
    def hand(self) -> list[str]:
        """Cartes qu'il a probablement en main : connues, et pas parmi les 4 dernières jouées."""
        last4 = self.played[-4:]
        return [c for c in self.deck if c not in last4]

    @property
    def next_in(self) -> str | None:
        """Prochaine carte à revenir dans sa main : celle jouée il y a 4 cartes."""
        return self.played[-4] if len(self.played) >= 4 else None

    def can_afford(self) -> list[str]:
        costs = {c: cost for c, cost, _ in UNIT2CARD.values()}
        return [c for c in self.hand if costs.get(c, 99) <= self.elixir]

    def summary(self) -> list[str]:
        lines = [f"ennemi : elixir ~{self.elixir:.1f}  deck connu {len(self.deck)}/8"]
        if self.deck:
            lines.append("deck : " + ", ".join(self.deck))
        if len(self.deck) >= 5:
            lines.append("main probable : " + (", ".join(self.hand) or "?"))
        if self.next_in:
            lines.append(f"revient bientot : {self.next_in}")
        if self.can_afford():
            lines.append("peut jouer : " + ", ".join(self.can_afford()))
        return lines
