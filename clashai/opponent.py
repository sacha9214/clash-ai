"""Modèle de l'adversaire : son élixir, son deck, sa main et ses prochaines cartes.

- Élixir : il commence à 5, gagne 1 toutes les 2,8 s (2x plus vite en fin de
  partie), plafonné à 10 ; chaque carte posée coûte son prix. On n'estime pas au
  dixième près (on ne voit pas tout), mais l'ordre de grandeur suffit pour punir
  un adversaire à sec.
- Cartes : une nouvelle unité ennemie (nouvel identifiant de suivi) née dans sa
  moitié et revue sur une autre image (pas seulement de mémoire) = une carte
  posée ; plusieurs unités identiques qui apparaissent ensemble (3 gobelins) = une
  seule carte. Une version évoluée / héros (« knight-evolution ») est la même carte.
  Ce qui sort d'une autre unité (Barbares du Bélier, Squelettes de la Sorcière…)
  n'est pas une carte jouée.
- Deck : les cartes avec des preuves (vues 2 fois, ou 1re pose très sûre), les plus
  vues d'abord (8 max) : une fausse détection du début ne verrouille pas le deck.
- Cycle : un deck fait 8 cartes, 4 en main ; la carte jouée passe derrière la
  file. Donc une carte jouée ne revient en main qu'après 4 autres cartes :
  main probable = deck connu moins les 4 dernières jouées ; la prochaine à
  rentrer en main = celle jouée il y a 4 cartes.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

# unité détectée (nom exact de la classe du détecteur, sans le camp « _0/_1 ») -> (carte, coût, nombre d'unités par carte)
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
    # classes du détecteur absentes jusqu'ici (coûts vérifiés dans data/cards/cards_db.json)
    "archer-queen": ("archer-queen", 5, 1), "battle-healer": ("battle-healer", 4, 1), "bowler": ("bowler", 5, 1),
    "cannon-cart": ("cannon-cart", 5, 1), "electro-dragon": ("electro-dragon", 5, 1),
    "electro-giant": ("electro-giant", 7, 1), "elixir-golem-big": ("elixir-golem", 3, 1),
    "fisherman": ("fisherman", 3, 1), "flying-machine": ("flying-machine", 4, 1),
    "golden-knight": ("golden-knight", 4, 1), "goblin-cage": ("goblin-cage", 4, 1),
    "goblin-drill": ("goblin-drill", 4, 1), "goblin-giant": ("goblin-giant", 6, 1),
    "heal-spirit": ("heal-spirit", 1, 1), "hunter": ("hunter", 4, 1), "lava-hound": ("lava-hound", 7, 1),
    "little-prince": ("little-prince", 3, 1), "magic-archer": ("magic-archer", 4, 1),
    "mighty-miner": ("mighty-miner", 4, 1), "monk": ("monk", 5, 1), "mother-witch": ("mother-witch", 4, 1),
    "night-witch": ("night-witch", 4, 1), "phoenix-big": ("phoenix", 4, 1), "royal-ghost": ("royal-ghost", 3, 1),
    "skeleton-dragon": ("skeleton-dragons", 4, 2), "skeleton-king": ("skeleton-king", 4, 1),
    "skeleton-barrel": ("skeleton-barrel", 3, 1), "sparky": ("sparky", 6, 1), "zappy": ("zappies", 4, 3),
    "goblin-demolisher": ("goblin-demolisher", 4, 1), "goblin-machine": ("goblin-machine", 5, 1),
    "suspicious-bush": ("suspicious-bush", 2, 1), "berserker": ("berserker", 2, 1),
    "rune-giant": ("rune-giant", 4, 1), "boss-bandit": ("boss-bandit", 6, 1),
    # sorts : visibles un court instant
    "arrows": ("arrows", 3, 1), "fireball": ("fireball", 4, 1), "zap": ("zap", 2, 1), "poison": ("poison", 4, 1),
    "the-log": ("the-log", 2, 1), "rocket": ("rocket", 6, 1), "lightning": ("lightning", 6, 1),
    "freeze": ("freeze", 4, 1), "earthquake": ("earthquake", 3, 1), "tornado": ("tornado", 3, 1),
    "goblin-barrel": ("goblin-barrel", 3, 1), "barbarian-barrel": ("barbarian-barrel", 2, 1),
    "giant-snowball": ("giant-snowball", 2, 1), "graveyard": ("graveyard", 5, 1), "rage": ("rage", 2, 1),
    "clone": ("clone", 3, 1), "royal-delivery": ("royal-delivery", 3, 1), "goblin-curse": ("goblin-curse", 2, 1),
    "vines": ("vines", 3, 1),
}
# classes « -evolution » / « -hero » du détecteur : même carte (et même coût) que la version de base
VARIANTS = """archer-evolution baby-dragon-evolution barbarian-barrel-hero barbarian-evolution bat-evolution
    battle-ram-evolution berserker-hero bomber-evolution dark-prince-hero dart-goblin-evolution
    electro-dragon-evolution executioner-evolution firecracker-evolution furnace-evolution giant-hero
    goblin-barrel-evolution goblin-cage-evolution goblin-drill-evolution goblin-giant-evolution goblin-hero
    hunter-evolution ice-golem-hero ice-spirit-evolution ice-wizard-hero inferno-dragon-evolution knight-evolution
    knight-hero lumberjack-evolution magic-archer-hero mega-minion-hero mini-pekka-hero mortar-evolution
    musketeer-evolution musketeer-hero princess-evolution royal-ghost-evolution royal-giant-evolution
    royal-hog-evolution royal-recruit-evolution skeleton-evolution tesla-evolution tombstone-hero
    valkyrie-evolution valkyrie-hero wall-breaker-evolution wizard-evolution wizard-hero zap-evolution""".split()


def base_name(name: str) -> str:
    """« knight-evolution », « giant-hero » -> « knight », « giant »."""
    for suffix in ("-evolution", "-hero"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


UNIT2CARD.update({v: UNIT2CARD[base_name(v)] for v in VARIANTS})
CARD_COST = {c: cost for c, cost, _ in UNIT2CARD.values()}
CARD_COST["goblin-gang"] = 3       # Gang de gobelins : Gobelins + Gobelins à lance posés ensemble (une seule carte)
GANG = {"goblins", "spear-goblins"}
SPELLS = {"arrows", "fireball", "zap", "poison", "the-log", "rocket", "lightning", "freeze", "earthquake",
          "tornado", "goblin-barrel", "barbarian-barrel", "giant-snowball", "graveyard", "rage", "clone",
          "royal-delivery", "goblin-curse", "vines"}
# unité (ou sort) -> ce qu'elle fait naître toute seule : ce ne sont pas des cartes jouées
SPAWNER_OF = {"goblin-hut": {"spear-goblin"}, "tombstone": {"skeleton"}, "goblin-cage": {"goblin-brawler"},
              "furnace": {"fire-spirit"}, "barbarian-hut": {"barbarian"},
              "battle-ram": {"barbarian"}, "barbarian-barrel": {"barbarian"}, "witch": {"skeleton"},
              "graveyard": {"skeleton"}, "skeleton-barrel": {"skeleton"}, "skeleton-king": {"skeleton"},
              "goblin-giant": {"spear-goblin"}, "goblin-barrel": {"goblin"}, "goblin-drill": {"goblin"},
              "suspicious-bush": {"goblin"}, "night-witch": {"bat"}, "lava-hound": {"lava-pup"},
              "golem": {"golemite"}, "elixir-golem-big": {"elixir-golem-mid", "elixir-golem-small"},
              "elixir-golem-mid": {"elixir-golem-small"}, "phoenix-big": {"phoenix-egg", "phoenix-small"},
              "phoenix-egg": {"phoenix-small"}, "little-prince": {"royal-guardian"},
              "royal-delivery": {"royal-recruit"}, "lumberjack": {"lumberjack-ghost", "rage"}}
# générateur -> sa carte (l'œuf du Phénix, le Golem d'élixir moyen sont eux-mêmes nés d'une carte)
SPAWNER_CARD = {b: UNIT2CARD[b][0] if b in UNIT2CARD else {"elixir-golem-mid": "elixir-golem",
                                                           "phoenix-egg": "phoenix"}[b] for b in SPAWNER_OF}
# unité -> cartes qui la font naître
SPAWNED_BY: dict[str, set[str]] = {}
for _b, _units in SPAWNER_OF.items():
    for _u in _units:
        SPAWNED_BY.setdefault(_u, set()).add(SPAWNER_CARD[_b])
# unités qui ne sont jamais une carte
SPAWNED = {"golemite", "lava-pup", "elixir-golem-mid", "elixir-golem-small", "phoenix-egg", "phoenix-small",
           "goblin-brawler", "royal-guardian", "lumberjack-ghost"}
REGEN, MAX_ELIXIR, DOUBLE_AFTER = 1 / 2.8, 10.0, 120.0
HIS_HALF, OUR_HALF = 0.43, 0.45     # fraction de hauteur : né chez lui / sort arrivé chez nous
STRONG, NOISE = 0.75, 0.6           # confiance : preuve forte (carte hors deck) / seuil de bruit
OUR_SPELL_S = 4.5                   # nos sorts (vol + effet + re-suivi) restent visibles jusqu'à 4,5 s après le tap
CYCLE_S = 6.0                       # même carte revue avant 6 s = la même pose, pas un nouveau cycle
SPAWN_ACTIVE_S = 10.0               # un générateur vu depuis moins de 10 s peut encore faire naître des unités


@dataclass
class _Candidate:
    """Suivi né chez lui (ou sort) pas encore jugé."""
    t0: float                          # première image
    x: float                           # position de naissance
    y: float
    spell: bool = False                # sort (attend 3 s d'arriver chez nous)
    n: int = 0                         # images où il a vraiment été vu (pas de mémoire)
    conf: float = 0.0                  # meilleure confiance vue
    near: frozenset = frozenset()      # unités que fait naître un générateur vu tout près à la naissance


@dataclass
class Opponent:
    start: float = field(default_factory=time.time)
    elixir: float = 5.0
    last_t: float = field(default_factory=time.time)
    played: list[str] = field(default_factory=list)       # cartes dans l'ordre joué
    seen_ids: set[int] = field(default_factory=set)       # suivis déjà vus (jugés ou en attente)
    recent: dict[str, tuple[float, int]] = field(default_factory=dict)   # carte -> (instant, unités vues)
    prev_counts: dict[str, int] = field(default_factory=dict)
    pending: dict[int, _Candidate] = field(default_factory=dict)
    spawner_seen: dict[str, float] = field(default_factory=dict)       # générateur -> dernier instant vu
    gained: float = 0.0
    our_spells: list[tuple] = field(default_factory=list)   # (instant, x, y, carte ou None) de nos sorts
    strong: set[str] = field(default_factory=set)         # cartes dont la 1re pose était sûre (conf >= 0.75)
    trusted: set[int] = field(default_factory=set)        # suivis crédibles : sûrs une fois, ou pose comptée
    memory: list[tuple] = field(default_factory=list)     # (instant, carte, x, y, suivi) ennemis des 3 dernières s
    spawner_pos: list[tuple] = field(default_factory=list)   # (instant, générateur, x, y) des 2 dernières s

    def note_our_spell(self, now: float, x: float, y: float, card: str | None = None) -> None:
        """Nous venons de lancer un sort (card : son nom) : ses images ne sont pas des cartes à lui."""
        self.our_spells = [s for s in self.our_spells if now - s[0] < OUR_SPELL_S + 1] + [(now, x, y, card)]

    def update(self, units, now: float | None = None, frame_h: int = 1280) -> list[str]:
        """Met à jour avec les unités détectées. Renvoie les cartes que l'ennemi vient de poser.

        Une carte posée = une unité qui APPARAÎT (nouvel identifiant de suivi) dans la
        moitié ennemie — on ne peut poser que chez soi. Le camp n'est lu qu'à la
        confirmation (le vote du détecteur peut changer sur les premières images) : elle
        doit être ennemie et vraiment revue (pas une copie de mémoire). Plusieurs unités
        du même type dans la même seconde et demie (3 gobelins) = une seule carte.
        Un sort ennemi est compté quand il arrive dans notre moitié (il part de sa tour),
        sauf si c'est le même sort que l'un des nôtres lancé il y a moins de 4,5 s."""
        now = time.time() if now is None else now
        rate = REGEN * (2 if now - self.start > DOUBLE_AFTER else 1)
        gain = (now - self.last_t) * rate
        self.gained += gain
        self.elixir = min(MAX_ELIXIR, self.elixir + gain)
        self.last_t = now
        fw = frame_h / 2.2                               # largeur de l'image (format du téléphone)
        # images réelles seulement : une copie de mémoire (coasted) ne confirme rien
        seen = [u for u in units if u.track_id >= 0 and not getattr(u, "coasted", False) and "tower" not in u.name]
        for u in units:
            b = base_name(u.name)
            if b in SPAWNER_OF:
                self.spawner_seen[b] = now
                self.spawner_pos.append((now, b, *u.center))
        self.spawner_pos = [p for p in self.spawner_pos if now - p[0] <= 2.0]
        for u in seen:
            if u.conf >= STRONG:
                self.trusted.add(u.track_id)
        # 1) nouveaux suivis : candidats s'ils naissent chez lui (sorts : partout), camp pas encore lu
        for u in seen:
            if u.track_id in self.seen_ids:
                continue
            self.seen_ids.add(u.track_id)
            x, y = u.center
            if base_name(u.name) in SPELLS or y < HIS_HALF * frame_h:
                near = frozenset(s for _, b, bx, by in self.spawner_pos
                                 if abs(bx - x) < 0.2 * fw and abs(by - y) < 0.12 * frame_h for s in SPAWNER_OF[b])
                self.pending[u.track_id] = _Candidate(now, x, y, spell=base_name(u.name) in SPELLS, near=near)
        # 2) juger les candidats revus sur cette image
        alive = {u.track_id: u for u in seen}
        new_cards = []
        for tid, c in list(self.pending.items()):
            u = alive.get(tid)
            if u is None:                            # pas revu (ou seulement de mémoire)
                if now - c.t0 > (3.0 if c.spell else 2.0):
                    del self.pending[tid]
                continue
            c.n, c.conf = c.n + 1, max(c.conf, u.conf)
            name = base_name(u.name)
            if name in SPELLS:
                # sort : jugé quand il arrive chez nous (un sort ennemi part de sa tour), sinon on attend 3 s
                if u.center[1] > OUR_HALF * frame_h and c.conf >= STRONG:
                    del self.pending[tid]
                    card = self._judge_spell(name, c, u, now, frame_h)
                else:
                    if now - c.t0 > 3.0:
                        del self.pending[tid]
                    continue
            elif c.n < 2 or not u.enemy:
                if now - c.t0 > 2.0:                 # le vote de camp peut encore basculer : on attend
                    del self.pending[tid]
                continue
            else:
                del self.pending[tid]
                card = self._judge_unit(tid, name, c, now, frame_h)
            if card:
                new_cards.append(card)
        self.memory = [m for m in self.memory if now - m[0] <= 3.0] + [
            (now, UNIT2CARD[b][0], *u.center, u.track_id) for u in units
            if u.enemy and (b := base_name(u.name)) in UNIT2CARD]
        if len(self.trusted) > 1000:
            self.trusted &= {u.track_id for u in units}
        return new_cards

    def _judge_unit(self, tid: int, name: str, c: _Candidate, now: float, frame_h: int) -> str | None:
        """Unité ennemie revue : nouvelle carte, ou sortie d'un générateur / re-suivi / bruit ?"""
        if name not in UNIT2CARD or name in SPAWNED or name in c.near:
            return None                              # pas une carte, ou sortie d'un générateur tout proche
        card = UNIT2CARD[name][0]
        if self._spawned(name, now):
            return None                              # un générateur de son deck est là : ce n'est pas une pose
        # le suivi perd parfois une unité et lui redonne un nouveau numéro : même carte ennemie vue tout près
        # (moins de ~3,5 cases) dans les 3 s AVANT sa naissance -> pas une nouvelle carte
        fw = frame_h / 2.2
        if any(mc == card and mt != tid and t < c.t0 and abs(mx - c.x) < 0.19 * fw and abs(my - c.y) < 0.06 * frame_h
               for t, mc, mx, my, mt in self.memory):
            return None
        t_last = self.recent.get(card, (-1e9, 0))[0]
        if now - t_last <= 1.5:                      # même pose (3 gobelins)
            self.trusted.add(tid)
            return None
        # règle du cycle, seulement si la carte vient d'être vue : sinon on a raté des poses entre deux
        if card in self.played[-4:] and now - t_last < CYCLE_S:
            return None
        deck = self.deck
        if card not in deck and c.conf < (STRONG if len(deck) >= 8 else NOISE):
            return None                              # carte inconnue peu sûre : bruit
        # il ne peut pas payer : estimation à ~0 et la carte coûte bien plus -> sans doute une unité déjà comptée
        # vue sous un autre nom (match du 27/09 : 35 cartes comptées, 5 poses en 9 s au début)
        if self.elixir < CARD_COST[card] - 2.5 and c.conf < STRONG:
            return None
        self.trusted.add(tid)
        return self._record(card, now, c.conf, heavy_resync=True)

    def _judge_spell(self, name: str, c: _Candidate, u, now: float, frame_h: int) -> str | None:
        """Sort arrivé chez nous et sûr : à lui, sauf si c'est le nôtre, s'il vient d'être compté ou s'il est lâché
        par une de ses unités."""
        card = UNIT2CARD[name][0]
        if now - self.recent.get(card, (-1e9, 0))[0] <= 3 or self._ours(card, *u.center, now, frame_h):
            return None
        if name in c.near or self._spawned(name, now):
            return None                              # Rage lâchée par son Bûcheron
        return self._record(card, now, c.conf, heavy_resync=False)

    def _ours(self, card: str, x: float, y: float, now: float, frame_h: int) -> bool:
        """Même sort que l'un des nôtres lancé il y a moins de 4,5 s (où qu'il soit), ou, carte inconnue,
        sort vu près de notre cible."""
        for s in self.our_spells:
            t, sx, sy = s[:3]
            ours = s[3] if len(s) > 3 else None
            if now - t > OUR_SPELL_S:
                continue
            if ours is None:
                if abs(sx - x) < 0.2 * frame_h / 2.2 and abs(sy - y) < 0.12 * frame_h:
                    return True
            elif UNIT2CARD.get(ours, (ours,))[0] == card:
                return True
        return False

    def _spawner_known(self, name: str) -> bool:
        """Une carte qui fait naître cette unité est dans son deck (ou vient d'être jouée) ?"""
        makers = SPAWNED_BY.get(name)
        if not makers:
            return False
        deck = self.deck
        return any(m in deck or self.last_t - self.recent.get(m, (-1e9, 0))[0] < 30 for m in makers)

    def _spawned(self, name: str, now: float) -> bool:
        """Unité née d'un générateur de son deck vu (ou joué) il y a peu : pas une carte posée."""
        if not self._spawner_known(name):
            return False
        return any(now - self.spawner_seen.get(b, -1e9) < SPAWN_ACTIVE_S
                   or now - self.recent.get(SPAWNER_CARD[b], (-1e9, 0))[0] < SPAWN_ACTIVE_S
                   for b, made in SPAWNER_OF.items() if name in made)

    def _record(self, card: str, now: float, conf: float, heavy_resync: bool) -> str:
        quiet = now - max((t for t, _ in self.recent.values()), default=self.start) > 5
        if card not in self.played and conf >= STRONG:
            self.strong.add(card)
        if card in GANG and self.played and self.played[-1] in GANG - {card}                 and now - self.recent.get(self.played[-1], (-1e9, 0))[0] < 1.5:
            # Gobelins puis Gobelins à lance (ou l'inverse) presque en même temps : c'est UN Gang de gobelins
            prev = self.played.pop()
            self.elixir += CARD_COST[prev]
            card = "goblin-gang"
        self.recent[card] = (now, 1)
        cost = CARD_COST[card]
        if heavy_resync and cost >= 6 and quiet:
            # carte lourde après un temps mort : il attendait d'être plein pour ne rien perdre
            # -> point de recalage fiable de l'estimation (il avait ~10)
            self.elixir = MAX_ELIXIR
        self.elixir = max(0.0, self.elixir - cost)
        self.played.append(card)
        return card

    @property
    def deck(self) -> list[str]:
        """Cartes connues de son deck (8 max) : celles avec des preuves (vues 2 fois, ou 1re pose sûre ;
        un sort vu une fois peut être du bruit), les plus vues d'abord, puis par ordre de 1re apparition."""
        order = list(dict.fromkeys(self.played))
        n = {c: self.played.count(c) for c in order}
        ok = [c for c in order if n[c] >= 2 or (c in self.strong and c not in SPELLS)]
        return sorted(ok, key=lambda c: -n[c])[:8]      # tri stable : à égalité, ordre d'apparition

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
        return [c for c in self.hand if CARD_COST.get(c, 99) <= self.elixir]

    def plausible(self, u) -> bool:
        """Une unité ennemie est-elle crédible ? Carte de son deck, unité qu'on ne peut pas vérifier (hors table,
        née d'une autre, sortie d'un générateur de son deck), suivi déjà crédible, ou détection très sûre."""
        if not u.enemy or "tower" in u.name or u.track_id in self.trusted:
            return True
        name = base_name(u.name)
        if name not in UNIT2CARD or name in SPAWNED or self._spawner_known(name):
            return True
        return UNIT2CARD[name][0] in self.deck or u.conf >= STRONG

    def banner(self) -> list[str]:
        """Trois lignes courtes pour le bandeau du haut."""
        known = len(self.deck)
        spent = sum(CARD_COST[c] for c in self.played)
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
