"""Cerveau v0 : des règles tactiques, en attendant l'apprentissage (phases 3-4).

Il regarde les unités détectées, sa main et son élixir, et répond comme un joueur
raisonnable : défendre ce qui arrive avec la bonne carte au bon endroit, sorts
sur les groupes (jamais sur du vide), et attaquer au Géant quand il a de l'élixir.

Toutes les positions sont en fractions de l'image (x vers la droite, y vers le bas).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from clashai.cards import AIR_UNITS, BUILDINGS, DECK, NOT_UNITS, SWARM_UNITS, TANK_UNITS

SPAWNERS = {"goblin-hut", "tombstone", "barbarian-hut", "furnace", "goblin-cage", "elixir-collector"}
from clashai.detect import Unit
from clashai.tiles import OWN_FIRST_ROW, PHONE

# Géométrie de l'arène (mesurée sur l'écran du REDMAGIC, flux 578x1280)
RIVER_Y = 0.425
LANES_X = (0.205, 0.795)
OWN_ZONE = (0.04, 0.96, 0.46, 0.695)       # où l'on peut poser une troupe (la clôture du fond est à 0.705)
OWN_TOWER_Y, KING_Y = 0.59, 0.66
# Temps entre la décision et l'effet visible du sort, MESURÉ en match (runs/games/*/spells.jsonl, agent.py) :
# Boule de feu 1.57-2.3 s (6 mesures, 5 à 23 cases), Flèches ~2.7 s (2 mesures). Il ne dépend presque pas
# de la distance -> constante par sort. À affiner avec plus de mesures.
SPELL_IMPACT_S = {"fireball": 2.0, "arrows": 2.6}
MAX_UNIT_SPEED_TILES = 3.0                  # au-delà, c'est du bruit de suivi
# Une unité s'ARRÊTE dès qu'une cible est à sa portée (nos troupes, nos tours) : la prédiction en tient compte.
# Portées en cases (valeurs du jeu, arrondies) ; mêlée par défaut.
ATTACK_RANGE_TILES = {"musketeer": 6.0, "archer": 5.0, "wizard": 5.5, "witch": 5.0, "princess": 9.0,
                      "dart-goblin": 6.5, "spear-goblin": 5.0, "electro-wizard": 5.0, "bomber": 4.5,
                      "executioner": 4.5, "baby-dragon": 3.5, "minion": 2.0, "mega-minion": 2.0,
                      "firecracker": 6.0, "magic-archer": 7.0, "hunter": 4.0, "ice-wizard": 5.5,
                      "royal-giant": 5.0, "flying-machine": 6.0, "inferno-dragon": 4.0}
MELEE_RANGE_TILES = 1.2
# Ne visent que les bâtiments : nos troupes ne les arrêtent pas
BUILDING_TARGETERS = {"giant", "golem", "royal-giant", "hog", "hog-rider", "balloon", "battle-ram", "ram-rider",
                      "electro-giant", "goblin-giant", "elixir-golem-big", "wall-breaker", "royal-hog"}
TOWER_HALF_TILES = 1.5                      # rayon approximatif d'une tour
ENEMY_TOWER_Y = 2 * RIVER_Y - OWN_TOWER_Y    # tours ennemies, symétriques des nôtres
CANNON_Y = PHONE.center(8, OWN_FIRST_ROW + 4)[1]   # 4 cases sous la rivière : le tank passe à portée des deux tours

# Coût des cartes ennemies, par unité détectée (pour juger l'échange d'élixir)
from clashai.opponent import UNIT2CARD  # noqa: E402
# Mêlée mono-cible : on la défend au centre, entre les deux tours, pour qu'elles tirent toutes les deux
SINGLE_MELEE = {"mini-pekka", "knight", "prince", "dark-prince", "lumberjack", "bandit", "pekka"}
# Cartes qui arrêtent bien notre Géant
GIANT_COUNTERS = {"mini-pekka", "pekka", "inferno-tower", "inferno-dragon", "cannon", "tesla", "skeleton-army",
                  "barbarians", "minion-horde", "prince", "goblin-gang", "guards"}
# Leurs grosses menaces : on garde un contre en main tant qu'elles peuvent revenir
WIN_CONDITIONS = {"giant", "hog-rider", "royal-giant", "golem", "pekka", "balloon", "battle-ram", "ram-rider",
                  "elixir-golem", "electro-giant", "goblin-giant", "giant-skeleton", "mega-knight"}


# Rayon réel des sorts dans le jeu (cases) et Roi ennemi (4x4 cases, colonnes 7-11, rangées 0.5-4.5).
# Toucher le Roi l'ACTIVE : il tire pour le reste du match -> nos sorts ne doivent jamais l'effleurer.
SPELL_RADIUS_TILES = {"arrows": 4.0, "fireball": 2.5}
ENEMY_KING_TILES = (7.0, 11.0, 0.5, 4.5)


def _tile_dist(ax: float, ay: float, bx: float, by: float) -> float:
    """Distance en cases entre deux points (fractions d'écran) : la caméra écrase les cases en hauteur."""
    return math.hypot((ax - bx) / PHONE.tw, (ay - by) / PHONE.th)


def _hits_enemy_king(card: str, x: float, y: float) -> bool:
    tx, ty = PHONE.to_tile(x, y)
    x0, x1, y0, y1 = ENEMY_KING_TILES
    dx, dy = max(x0 - tx, 0, tx - x1), max(y0 - ty, 0, ty - y1)
    return math.hypot(dx, dy) < SPELL_RADIUS_TILES.get(card, 3.0) + 0.3   # petite marge d'imprécision


# ---- Adaptation au deck adverse (cartes vues par opponent.py) ----
FAST_BUILDING_HUNTERS = {"hog-rider", "hog", "battle-ram", "ram-rider", "royal-hog"}
SLOW_TANKS = {"giant", "golem", "giant-skeleton", "electro-giant", "goblin-giant", "elixir-golem-big"}
BIG_SPELLS = {"fireball", "poison", "lightning", "rocket", "earthquake"}
SMALL_SPELLS = {"arrows", "zap", "the-log", "giant-snowball", "barbarian-barrel", "tornado"}
PULL_BUILDINGS = {"cannon", "tesla", "inferno-tower", "bomb-tower", "goblin-cage", "tombstone"}


def _cost(unit_name: str) -> int:
    v = UNIT2CARD.get(unit_name)
    return v[1] if v else 3


@dataclass
class Decision:
    card: str
    slot: int
    x: float
    y: float
    reason: str
    tile: tuple[int, int] | None = None   # case de la grille 18x32 où la carte est posée


@dataclass
class Seen:
    name: str
    enemy: bool
    x: float        # centre (fraction)
    y: float        # pieds (bas de la boîte)
    vx: float = 0.0  # vitesse en fraction/s (depuis le suivi)
    vy: float = 0.0


# Emprises de nos tours (centre x, centre y, demi-largeur, demi-hauteur) : on ne peut
# rien poser dessus — le jeu refuserait et la carte resterait sélectionnée.
OWN_TOWERS = ((0.205, 0.59, 0.085, 0.05), (0.795, 0.59, 0.085, 0.05), (0.5, 0.665, 0.115, 0.065))
OWN_TOWER_CENTERS = ((0.205, 0.59), (0.795, 0.59), (0.5, 0.66))


def _clamp_own(x: float, y: float) -> tuple[float, float]:
    x0, x1, y0, y1 = OWN_ZONE
    x, y = min(max(x, x0), x1), min(max(y, y0), y1)
    for cx, cy, hw, hh in OWN_TOWERS:
        if abs(x - cx) < hw and abs(y - cy) < hh:
            # sortir par le bord le plus proche (jamais vers le haut du terrain si possible)
            dx, dy = hw - abs(x - cx), hh - abs(y - cy)
            if dx < dy:
                x = cx + math.copysign(hw + 0.01, x - cx or 1)
            else:
                y = cy + math.copysign(hh + 0.01, y - cy or 1)
    return min(max(x, x0), x1), min(max(y, y0), y1)


def _lane(x: float) -> int:
    return 0 if x < 0.5 else 1


class Brain:
    def __init__(self, params: dict | None = None):
        from clashai.strategy import DEFAULT
        self.p = dict(DEFAULT, **(params or {}))
        self.giant_lane: int | None = None
        self.giant_time = -1e9
        self.last_defense_lane: int | None = None
        self.opp_elixir = 5.0        # estimation fournie par le modèle de l'adversaire
        self.opp_hand: list[str] = []            # sa main probable (cartes connues hors des 4 dernières)
        self.opp_deck: list[str] = []
        self.opp_heavy_t = -1e9                  # instant de sa dernière carte à 6+ élixir
        self._ours: list[Seen] = []
        self.match = None                        # towers.MatchState : PV de nos tours, tours ennemies, phase
        self.last_own_play: tuple[int, float] | None = None   # (couloir, instant) de notre dernière troupe
        self.own_recent: list[tuple[tuple[str, ...], float, float, float]] = []   # (unités, x, y, instant) posées par nous

    # ---- perception -> monde simplifié ----
    @staticmethod
    def to_seen(units: list[Unit], w: int, h: int, trails: dict | None, fps: float) -> list[Seen]:
        out = []
        for u in units:
            if (u.name in SPAWNERS or u.name == "goblin-barrel") and u.enemy:
                out.append(Seen(u.name, True, (u.box[0] + u.box[2]) / 2 / w, (u.box[1] + u.box[3]) / 2 / h))
                continue
            if u.name in BUILDINGS or u.name in NOT_UNITS:
                continue
            x, y = (u.box[0] + u.box[2]) / 2 / w, u.box[3] / h
            vx = vy = 0.0
            tr = trails.get(u.track_id) if trails else None
            if tr and len(tr) >= 4 and fps > 0:
                (ax, ay), (bx, by) = tr[-4], tr[-1]
                dt = 3 / fps
                vx, vy = (bx - ax) / w / dt, (by - ay) / h / dt
            out.append(Seen(u.name, u.enemy, x, y, vx, vy))
        return out

    # ---- décision ----
    def decide(self, seen: list[Seen], hand: list[str | None], ready: list[bool], elixir: float,
               now: float) -> Decision | None:
        # le détecteur prend parfois NOTRE unité fraîchement posée pour une ennemie (ex. notre Géant
        # « ennemi » sur la case où on vient de le poser) : on la remet dans notre camp
        self.own_recent = [r for r in self.own_recent if now - r[3] < 6]
        seen = [Seen(s.name, False, s.x, s.y, s.vx, s.vy)
                if s.enemy and s.y > RIVER_Y and any(s.name in names and _tile_dist(s.x, s.y, x, y) < 3
                                                     for names, x, y, _ in self.own_recent) else s
                for s in seen]
        self._ours = [s for s in seen if not s.enemy]
        d = self._decide(seen, hand, ready, elixir, now)
        if d is None:
            return None
        if DECK[d.card].kind != "spell":
            # le jeu pose au centre d'une case : on vise ce centre (et on reste hors des tours)
            d.x, d.y = PHONE.snap(*_clamp_own(*PHONE.snap(d.x, d.y)))
            self.own_recent.append((DECK[d.card].units, d.x, d.y, now))
            if abs(d.x - 0.5) > 0.06:                                  # pas une carte posée au centre
                self.last_own_play = (_lane(d.x), now)
        d.tile = PHONE.cell(d.x, d.y)
        return d

    def _decide(self, seen: list[Seen], hand: list[str | None], ready: list[bool], elixir: float,
                now: float) -> Decision | None:
        playable = {c: i for i, c in enumerate(hand) if c in DECK and ready[i] and DECK[c].cost <= elixir + 0.3}
        enemies = [s for s in seen if s.enemy]
        # Tonneau à gobelins en vol : Valkyrie juste derrière la tour visée, elle balaie les 3 gobelins à l'atterrissage
        barrel = next((e for e in enemies if e.name == "goblin-barrel"), None)
        if barrel and "valkyrie" in playable:
            lane_x = LANES_X[_lane(barrel.x)]
            x, y = _clamp_own(lane_x, OWN_TOWER_Y + 0.06)
            return Decision("valkyrie", playable["valkyrie"], x, y, "défense : Tonneau à gobelins -> Valkyrie derrière la tour")
        enemies = [e for e in enemies if e.name != "goblin-barrel"]
        threats = [s for s in enemies if s.y > RIVER_Y - self.p['defend_line']]   # sur notre moitié ou au pont

        d = self._spells([e for e in enemies if e.name not in SPAWNERS], playable)
        if d:
            return d
        # bâtiment qui produit des unités sans arrêt : une Boule de feu le rentabilise
        spawners = [e for e in enemies if e.name in SPAWNERS]
        if spawners and self.p.get("fireball_spawners") and "fireball" in playable and elixir >= 7:
            t = spawners[0]
            return Decision("fireball", playable["fireball"], t.x, t.y, f"fireball sur {t.name}")
        enemies = [e for e in enemies if e.name not in SPAWNERS]
        # ne pas surinvestir : une menace déjà couverte par nos unités est ignorée (sauf un tank)
        ours = [s for s in seen if not s.enemy]
        threats = [t for t in threats if t.name in TANK_UNITS or
                   sum(math.hypot(o.x - t.x, o.y - t.y) < 0.15 for o in ours) <
                   sum(math.hypot(e.x - t.x, e.y - t.y) < 0.15 for e in threats)]
        if self.p.get("ignore_small") and not any(self._tower_low(_lane(t.x)) for t in threats):
            # 1-2 petites unités (gobelin, squelette…) : les tours s'en chargent, on garde l'élixir
            # (sauf si la tour de ce couloir est basse : là, chaque point de vie compte)
            small = [t for t in threats if t.name in SWARM_UNITS]
            if len(small) == len(threats) and len(small) <= 2:
                threats = []
            # une unité seule à 1-2 élixir : nos défenseurs coûtent 3+, la tour encaisse, on gagne l'échange
            elif len(threats) == 1 and threats[0].name not in TANK_UNITS and _cost(threats[0].name) <= 2:
                threats = []
        if threats:
            d = self._defend(threats, playable)
            if d:
                self.last_defense_lane = _lane(d.x)
            return d
        # Canon posé à l'avance : son tank ou son Cochon descend vers le pont (38 % des Canons des pros)
        coming = [e for e in enemies if (e.name in TANK_UNITS or e.name in FAST_BUILDING_HUNTERS) and e.name not in AIR_UNITS
                  and RIVER_Y - 6 * PHONE.th < e.y <= RIVER_Y and e.vy >= 0]
        if coming and "cannon" in playable:
            t = max(coming, key=lambda e: e.y)
            row = OWN_FIRST_ROW + (4 if t.name in FAST_BUILDING_HUNTERS else 6 if t.name in SLOW_TANKS else 3)
            x, y = PHONE.center(8 if t.x < 0.5 else 9, row)
            return Decision("cannon", playable["cannon"], x, y, f"canon à l'avance : {t.name} arrive vers le pont")
        # garder le contre de sa grosse menace (Géant, Cochon…) tant qu'elle peut revenir dans sa main
        keep = set()
        if set(self.opp_hand) & WIN_CONDITIONS:
            keep |= {"cannon", "mini-pekka"}
        if "goblin-barrel" in self.opp_hand:
            keep.add("valkyrie")            # le seul contre propre au Tonneau
        playable = {c: i for c, i in playable.items() if c not in keep}
        return self._attack(seen, playable, elixir, now)

    def _spells(self, enemies: list[Seen], playable: dict) -> Decision | None:
        for card, min_count, kinds in (("arrows", self.p["arrows_min"], SWARM_UNITS),
                                       ("fireball", self.p["fireball_min"], None)):
            if card not in playable:
                continue
            pool = [e for e in enemies if kinds is None or e.name in kinds]
            if card == "fireball":
                pool = [e for e in pool if e.name not in SWARM_UNITS or e.name in ("barbarian", "archer")]
            r = DECK[card].radius / PHONE.tw          # rayon de regroupement, en cases
            best, center = 0, None
            for e in pool:
                # où sera chaque unité quand le sort tombera sur e ?
                t = self._impact_time(card, *self._future(e, 0))
                ex, ey = self._future(e, t)
                group = [o for o in pool if _tile_dist(*self._future(o, t), ex, ey) < r]
                c = (sum(self._future(o, t)[0] for o in group) / len(group),
                     sum(self._future(o, t)[1] for o in group) / len(group))
                # on recalcule l'impact pour le centre visé et on ne compte que les unités encore DEDANS
                # (marge de 15 %) : celles qui sortent du cercle en marchant ne comptent pas
                t = self._impact_time(card, *c)
                inside = [o for o in group if _tile_dist(*self._future(o, t), *c) < 0.85 * r]
                if len(inside) > best:
                    if _hits_enemy_king(card, *c):
                        continue   # ce groupe est collé au Roi ennemi : on ne l'active pas pour quelques troupes
                    best, center = len(inside), c
            if card == "fireball" and best < 2 and self.p.get("fireball_patient") and center and not any(
                    _tile_dist(center[0], center[1], lx, ENEMY_TOWER_Y) < r for lx in LANES_X):
                continue   # cible seule loin d'une tour : on attend qu'elle s'en approche ou qu'une 2e la rejoigne
            if best >= min_count and (best >= 2 or card != "fireball" or
                                      any(e.name in TANK_UNITS or e.name in ("musketeer", "wizard", "witch",
                                          "executioner", "princess", "dart-goblin") for e in pool)):
                return Decision(card, playable[card], *center, f"{card} sur un groupe de {best}")
        return None

    def _future(self, u: Seen, t: float) -> tuple[float, float]:
        """Position de l'unité dans t secondes : elle avance (vitesse du suivi, bornée contre le bruit)
        jusqu'à ce qu'une de nos troupes ou tours soit à sa portée, puis elle s'arrête pour se battre."""
        vx, vy = u.vx / PHONE.tw, u.vy / PHONE.th          # cases/s
        v = math.hypot(vx, vy)
        if v <= 0 or t <= 0:
            return u.x, u.y
        k = min(1.0, MAX_UNIT_SPEED_TILES / v)
        rng = ATTACK_RANGE_TILES.get(u.name, MELEE_RANGE_TILES)
        melee = rng <= MELEE_RANGE_TILES + 0.1
        blockers = [(bx, by, TOWER_HALF_TILES) for bx, by in OWN_TOWER_CENTERS]
        if u.name not in BUILDING_TARGETERS:
            blockers += [(o.x, o.y, 0.5) for o in self._ours
                         if not (melee and o.name in AIR_UNITS)]        # la mêlée ne touche pas les volants
        x, y, step = u.x, u.y, 0.1
        for _ in range(int(t / step) + 1):
            if any(_tile_dist(x, y, bx, by) <= rng + size for bx, by, size in blockers):
                break                                               # une cible à portée : elle s'arrête là
            x, y = x + u.vx * k * step, y + u.vy * k * step
        return x, y

    @staticmethod
    def _impact_time(card: str, x: float, y: float) -> float:
        return SPELL_IMPACT_S.get(card, 2.0)

    def _defend(self, threats: list[Seen], playable: dict) -> Decision | None:
        # la menace la plus proche de nos tours (le plus bas à l'écran)
        t = max(threats, key=lambda s: s.y)
        lane_x = LANES_X[_lane(t.x)]
        is_air, is_tank = t.name in AIR_UNITS, t.name in TANK_UNITS
        swarm = sum(1 for s in threats if s.name in SWARM_UNITS and math.hypot(s.x - t.x, s.y - t.y) < 0.15)
        if is_air:
            order = ["musketeer", "archers", "minions", "spear-goblins"]
        elif is_tank:
            order = ["mini-pekka", "musketeer", "knight", "valkyrie", "minions", "archers"]
        elif swarm >= 2:
            order = ["valkyrie", "knight", "musketeer", "archers", "mini-pekka"]   # dégâts de zone
        else:
            order = ["knight", "valkyrie", "mini-pekka", "musketeer", "archers", "minions"]
        # Canon : le bâtiment au centre attire les tanks (Géant, Hog…) entre les deux tours
        if "cannon" in playable and not is_air and (is_tank or t.name in FAST_BUILDING_HUNTERS or t.name in SINGLE_MELEE
                                                     or swarm >= 2 or _cost(t.name) >= 3):
            col = 8 if lane_x < 0.5 else 9             # centre, côté de la menace
            if t.name in FAST_BUILDING_HUNTERS:
                row, why = OWN_FIRST_ROW + 4, "4 cases sous la rivière : le Cochon est dévié entre les 2 tours"
                if set(self.opp_hand) & BIG_SPELLS:
                    # il a Boule de feu/Poison en main : il visera l'emplacement habituel -> on le pose plus haut
                    row, why = OWN_FIRST_ROW + 2, "plus haut que d'habitude : son sort prédit tombera à côté"
            elif t.name == "royal-giant":
                row, why = OWN_FIRST_ROW + 2, "près de la rivière : le Géant royal tire de loin, il faut l'attirer tôt"
            elif t.name in SLOW_TANKS:
                row, why = OWN_FIRST_ROW + 6, "en retrait : tank lent, long chemin sous le feu des 2 tours"
            else:
                row, why = OWN_FIRST_ROW + 4, "4 cases sous la rivière"
            x, y = PHONE.center(col, row)
            return Decision("cannon", playable["cannon"], x, y, f"défense : {t.name} -> canon ({why})")
        # échange d'élixir : parmi les cartes adaptées, ne pas payer plus que l'attaque (+1) si une moins chère suffit
        push_cost = sum(_cost(n) for n in {s.name for s in threats if math.hypot(s.x - t.x, s.y - t.y) < 0.2})
        fits = [c for c in order if c in playable and c in DECK]
        cheap = [c for c in fits if DECK[c].cost <= push_cost + 1]
        for card in cheap or sorted(fits, key=lambda c: DECK[c].cost):   # sinon, au moins la moins chère
            c = DECK[card]
            if t.name in SINGLE_MELEE and c.kind == "troop" and c.targets == "ground" and t.y < OWN_TOWER_Y - 0.04:
                # mêlée mono-cible : au centre entre les tours, l'ennemi dévie vers le milieu et les deux tours tirent
                x, y = _clamp_own(0.5 + (-0.04 if lane_x < 0.5 else 0.04), OWN_TOWER_Y - 0.03)
                return Decision(card, playable[card], x, y, f"défense : {t.name} -> {card} au centre (les 2 tours tirent)")
            if c.targets == "air+ground" and not c.flying:
                # tireur : derrière la tour, décalé vers le centre -> la tour et lui tirent ensemble
                x, y = lane_x + (0.14 if lane_x < 0.5 else -0.14), OWN_TOWER_Y + 0.05
                if set(self.opp_hand) & BIG_SPELLS:
                    # sa Boule de feu toucherait tour + tireur : on le recule (~5 cases derrière la tour, hors du rayon)
                    y = OWN_TOWER_Y + 0.085
            elif t.y < RIVER_Y + 0.05:
                # l'ennemi est encore au pont : on pose juste devant la tour pour l'attirer
                x, y = lane_x + (0.08 if lane_x < 0.5 else -0.08), OWN_TOWER_Y - 0.06
            else:
                # au contact : sur le chemin de l'unité, un peu en retrait
                x, y = t.x + t.vx * 0.5, t.y + 0.05
            x, y = _clamp_own(x, y)
            kind = "volante" if is_air else "tank" if is_tank else "au sol"
            return Decision(card, playable[card], x, y, f"défense : {t.name} ({kind}) -> {card}")
        return None

    def _tower_low(self, lane: int) -> bool:
        return self.match is not None and self.match.our_hp[lane] < 0.35

    @staticmethod
    def _enemy_lane(seen: list[Seen]) -> int | None:
        """Le côté où joue l'ennemi : celui de son unité la plus avancée vers nous, sinon celui où il en a le plus."""
        enemies = [s for s in seen if s.enemy]
        if not enemies:
            return None
        return _lane(max(enemies, key=lambda s: s.y).x)

    def _attack_lane(self, seen: list[Seen], now: float | None = None) -> int:
        """Où jouer nos troupes. Vidéos (39 vidéos, 447 coups de pros avec nos cartes) : ils jouent du côté de
        l'ennemi le plus avancé dans 76-86 % des cas, Géant compris. Avant tout : une tour ennemie tombée."""
        if self.match is not None:
            dead = [l for l in (0, 1) if not self.match.enemy_alive[l]]
            if len(dead) == 1:
                return dead[0]
        lane = self._enemy_lane(seen)
        if lane is not None:
            return lane
        if now is not None and self.last_own_play and now - self.last_own_play[1] < 15:
            return self.last_own_play[0]            # rien en face : on reste avec nos troupes
        return self._weak_lane(seen)

    def _attack(self, seen: list[Seen], playable: dict, elixir: float, now: float) -> Decision | None:
        fast = self.match is not None and self.match.phase(now) != "normal"   # l'élixir revient 2x plus vite
        # punir : l'adversaire vient de dépenser, il ne peut pas bien défendre tout de suite
        punish = self.p.get("punish_low_elixir") and self.opp_elixir < 3 and elixir >= 5
        # il vient de poser une carte lourde : il est à sec, on frappe tout de suite dans l'AUTRE couloir
        if self.p.get("punish_opposite") and now - self.opp_heavy_t < 4 and elixir >= 4:
            for card in ("mini-pekka", "knight"):
                if card in playable:
                    lane = self._weak_lane(seen)
                    x, y = _clamp_own(LANES_X[lane], 0.47)
                    self.opp_heavy_t = -1e9
                    return Decision(card, playable[card], x, y, f"punition : carte lourde en face -> {card} au pont opposé")
        giant_at = self.p["giant_elixir"] - (1 if fast else 0)
        # ses contres au Géant connus et tous hors de sa main (joués récemment) : fenêtre pour lancer plus tôt
        known = set(self.opp_deck) & GIANT_COUNTERS
        if self.p.get("giant_when_counter_out") and known and not known & set(self.opp_hand):
            giant_at -= 2
        if "giant" in playable and (elixir >= giant_at or punish):
            lane = self._attack_lane(seen, now if self.p["counter_push"] else None)
            tower_down = self.match is not None and not all(self.match.enemy_alive.values())
            self.giant_lane, self.giant_time = lane, now
            spot = self.p["giant_spot"]
            if spot == "back" and set(self.opp_deck) & PULL_BUILDINGS:
                spot = "corner"     # son bâtiment ne pourra pas tirer le Géant vers le centre depuis le coin
            lx = LANES_X[lane] + ((-0.12 if lane == 0 else 0.12) if spot == "corner" else 0)
            if spot == "king":
                lx = 0.5 + (-0.12 if lane == 0 else 0.12)   # juste derrière le Roi, du côté du couloir visé
            x, y = _clamp_own(lx, {"king": 0.69, "back": 0.69, "corner": 0.69, "mid": 0.55, "bridge": 0.47}.get(spot, 0.69))
            countered_out = self.p.get("giant_when_counter_out") and known and not known & set(self.opp_hand)
            why = "l'ennemi est à sec" if punish and elixir < giant_at else \
                "ses contres sont joués" if countered_out and elixir < self.p["giant_elixir"] else \
                "double élixir" if fast and elixir < self.p["giant_elixir"] else \
                "sa tour de ce côté est tombée" if tower_down and lane == self._attack_lane(seen) else \
                {"back": "au fond", "corner": "dans le coin" + (" (il a un bâtiment)" if set(self.opp_deck) & PULL_BUILDINGS else ""),
                 "mid": "au milieu", "bridge": "au pont", "king": "derrière le Roi"}.get(spot, spot)
            if punish and elixir < giant_at:
                x, y = _clamp_own(LANES_X[lane], 0.47)   # pression immédiate au pont
            return Decision("giant", playable["giant"], x, y, f"attaque : Géant ({why})")
        # soutien derrière notre Géant pendant qu'il avance
        ours = [s for s in seen if not s.enemy and s.name == "giant"]
        if ours and elixir >= self.p["support_min_elixir"]:
            g = ours[0]
            order = ["musketeer", "archers", "valkyrie", "mini-pekka", "minions"]
            if set(self.opp_hand) & SMALL_SPELLS:
                # ses Flèches/Zap/Bûche tueraient archères ou gargouilles : on les passe en dernier
                order = [c for c in order if c not in ("archers", "minions")] + ["archers", "minions"]
            for card in order:
                if card in playable:
                    x, y = _clamp_own(g.x, g.y + 0.07)       # ~4 cases derrière : hors d'une Boule de feu sur le Géant
                    return Decision(card, playable[card], x, y, f"soutien : {card} derrière le Géant")
        if elixir >= self.p["cycle_at"] - (1.5 if fast else 0) and playable:
            # élixir plein et rien à faire : on fait tourner la carte la moins chère, sans risque
            card = min((c for c in playable if DECK[c].kind == "troop"), key=lambda c: DECK[c].cost, default=None)
            if card == "archers":
                # au centre juste devant le Roi : les deux archères partent chacune dans un couloir,
                # une seule Flèche ne peut plus les prendre ensemble
                x, y = _clamp_own(0.5, OWN_TOWER_Y + 0.0)
                return Decision(card, playable[card], x, y, "rien à faire : archères séparées au centre")
            if card in ("mini-pekka", "knight", "valkyrie"):
                # les pros les jouent dans un couloir, rangées 19-25 (vidéos), jamais derrière le Roi
                lane = self._attack_lane(seen, now)
                x, y = PHONE.center(3 if lane == 0 else 14, OWN_FIRST_ROW + 5)
                return Decision(card, playable[card], x, y, f"élixir plein : {card} dans le couloir, prêt à contre-attaquer")
            if card:
                x, y = _clamp_own(0.5 + (0.1 if self._weak_lane(seen) else -0.1), 0.69)
                return Decision(card, playable[card], x, y, f"élixir plein : {card} derrière le Roi")
        return None

    @staticmethod
    def _weak_lane(seen: list[Seen]) -> int:
        """Le couloir où l'ennemi a le moins d'unités (on attaque là où il ne défend pas)."""
        cnt = [0, 0]
        for s in seen:
            if s.enemy:
                cnt[_lane(s.x)] += 1
        return 0 if cnt[0] <= cnt[1] else 1
