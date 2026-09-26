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
# bâtiment -> unités qu'il produit tout seul (ce ne sont pas des cartes jouées)
SPAWNER_OF = {"goblin-hut": {"spear-goblin"}, "tombstone": {"skeleton"}, "goblin-cage": {"goblin-brawler"},
              "furnace": {"fire-spirit"}, "barbarian-hut": {"barbarian"}}
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
    pending: dict[int, tuple[str, float]] = field(default_factory=dict)
    spawner_seen: dict[str, float] = field(default_factory=dict)
    gained: float = 0.0
    our_spells: list[tuple[float, float, float]] = field(default_factory=list)   # (instant, x, y) de nos sorts

    def note_our_spell(self, now: float, x: float, y: float) -> None:
        self.our_spells = [s for s in self.our_spells if now - s[0] < 5] + [(now, x, y)]

    def update(self, units, now: float | None = None, frame_h: int = 1280) -> list[str]:
        """Met à jour avec les unités détectées. Renvoie les cartes que l'ennemi vient de poser.

        Une carte posée = une unité qui APPARAÎT (nouvel identifiant de suivi) dans la
        moitié ennemie — on ne peut poser que chez soi, donc tout ce qui naît là-haut
        est à lui, quel que soit le camp que le détecteur a deviné. Elle doit être revue
        sur l'analyse suivante (sinon c'est du bruit). Plusieurs unités du même type
        dans la même seconde et demie (3 gobelins) = une seule carte.
        Les sorts sont ignorés (effets trop brefs, confondus avec les nôtres)."""
        now = time.time() if now is None else now
        rate = REGEN * (2 if now - self.start > DOUBLE_AFTER else 1)
        gain = (now - self.last_t) * rate
        self.gained += gain
        self.elixir = min(MAX_ELIXIR, self.elixir + gain)
        self.last_t = now
        alive = {u.track_id: u for u in units if u.track_id >= 0}
        new_cards = []
        # 1) confirmer les candidats de l'analyse précédente encore présents
        for tid, (card, t0) in list(self.pending.items()):
            if tid in alive:
                del self.pending[tid]
                t_last = self.recent.get(card, (-1e9, 0))[0]
                # règle du cycle : une carte jouée ne revient en main qu'après 4 autres cartes
                if card in self.played[-4:]:
                    continue
                if len(self.deck) >= 8 and card not in self.deck:
                    continue                         # deck complet : c'est une fausse détection
                if now - t_last > 1.5:
                    quiet = now - max((t for t, _ in self.recent.values()), default=self.start) > 5
                    self.recent[card] = (now, 1)
                    _, cost, _ = next(v for v in UNIT2CARD.values() if v[0] == card)
                    if cost >= 6 and quiet:
                        # carte lourde après un temps mort : il attendait d'être plein pour ne rien perdre
                        # -> point de recalage fiable de l'estimation (il avait ~10)
                        self.elixir = MAX_ELIXIR
                    self.elixir = max(0.0, self.elixir - cost)
                    self.played.append(card)
                    new_cards.append(card)
            elif now - t0 > 2.0:
                del self.pending[tid]
        # 2) nouveaux candidats : unités nées dans la moitié ennemie
        prev = getattr(self, "_prev_pos", [])
        spawners = [(u.name, u.center) for u in units if u.name in SPAWNER_OF]
        for u in units:
            if u.name in SPAWNER_OF:
                self.spawner_seen[u.name] = now
        for tid, u in alive.items():
            if tid in self.seen_ids:
                continue
            self.seen_ids.add(tid)
            if u.name in SPELLS and u.name in UNIT2CARD:
                mine = any(now - t < 3 and abs(x - u.center[0]) < 0.2 * frame_h / 2.2 and abs(y - u.center[1]) < 0.12 * frame_h
                           for t, x, y in self.our_spells)
                last = self.recent.get(u.name, (-1e9, 0))[0]
                ours_half = u.center[1] > 0.45 * frame_h      # un sort ennemi vise nos unités/tours
                if not mine and ours_half and u.conf >= 0.75 and now - last > 3:
                    self.recent[u.name] = (now, 1)
                    self.elixir = max(0.0, self.elixir - UNIT2CARD[u.name][1])
                    self.played.append(u.name)
                    new_cards.append(u.name)
                continue
            if u.name in SPAWNED or u.name not in UNIT2CARD:
                continue
            card = UNIT2CARD[u.name][0]
            # sortie d'un bâtiment (Cabane -> Gobelins à lance, Pierre tombale -> Squelettes…)
            if any(u.name in SPAWNER_OF[b] and abs(bx - u.center[0]) < 0.2 * frame_h / 2.2
                   and abs(by - u.center[1]) < 0.12 * frame_h for b, (bx, by) in spawners):
                continue
            if any(u.name in SPAWNER_OF[b] and now - t < 6 for b, t in self.spawner_seen.items()) \
                    and self.played.count(card) >= 1 and now - self.recent.get(card, (-1e9, 0))[0] < 12:
                continue
            # le suivi perd parfois une unité et lui redonne un nouveau numéro : si une unité
            # de la même carte était là, tout près, à l'analyse précédente, ce n'est pas une nouvelle carte
            if any(c == card and abs(x - u.center[0]) < 0.12 * frame_h / 2.2 and abs(y - u.center[1]) < 0.08 * frame_h
                   for c, x, y in prev):
                continue
            if u.center[1] < 0.43 * frame_h:
                self.pending[tid] = (card, now)
        self._prev_pos = [(UNIT2CARD[u.name][0], u.center[0], u.center[1]) for u in units
                          if u.name in UNIT2CARD and u.enemy]
        return new_cards

    @property
    def deck(self) -> list[str]:
        """Cartes connues de son deck, dans l'ordre de première apparition (8 max)."""
        out = []
        for c in self.played:
            if c in out or (c in SPELLS and self.played.count(c) < 2):   # un sort vu une fois peut être du bruit
                continue
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

    def plausible(self, u) -> bool:
        """Une unité ennemie est-elle crédible ? Carte de son deck connu, ou détection très sûre (nouvelle carte).
        Deck complet (8 cartes vues) : rien d'autre n'est possible."""
        if not u.enemy or "tower" in u.name:
            return True
        if u.name not in UNIT2CARD:                  # unité hors de la table (Cage, Soigneuse…) : seulement si sûre
            return u.conf >= 0.75 and len(self.deck) < 8
        card = UNIT2CARD[u.name][0]
        if card in self.deck:
            return True
        return len(self.deck) < 8 and u.conf >= 0.75

    def banner(self) -> list[str]:
        """Trois lignes courtes pour le bandeau du haut."""
        known = len(self.deck)
        spent = sum(next(v[1] for v in UNIT2CARD.values() if v[0] == c) for c in self.played)
        l1 = (f"ADVERSAIRE elixir {self.elixir:.1f}/10 = 5 + {self.gained:.0f} gagnes - {spent} joues"
              f"   ({known}/8 cartes)")
        l2 = "deck : " + (", ".join(self.deck) or "?")
        l3 = ("main probable : " + (", ".join(self.hand) or "?")) if known >= 5 else "main : attendre 5 cartes vues"
        if self.next_in:
            l3 += f"  | revient : {self.next_in}"
        return [l1, l2, l3]

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
