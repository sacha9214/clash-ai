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
# Temps avant l'impact : le sort part de notre Roi et vole jusqu'à la cible, plus le temps de poser la
# carte (play_ms médian 160 ms + affichage). Vitesses en cases/s : estimations, à caler sur des matchs.
ACTION_DELAY_S = 0.25
SPELL_SPEED_TILES = {"fireball": 10.0, "arrows": 14.0}
MAX_UNIT_SPEED_TILES = 3.0                  # au-delà, c'est du bruit de suivi
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
        d = self._decide(seen, hand, ready, elixir, now)
        if d is None:
            return None
        if DECK[d.card].kind != "spell":
            # le jeu pose au centre d'une case : on vise ce centre (et on reste hors des tours)
            d.x, d.y = PHONE.snap(*_clamp_own(*PHONE.snap(d.x, d.y)))
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
        if self.p.get("ignore_small"):
            # 1-2 petites unités (gobelin, squelette…) : les tours s'en chargent, on garde l'élixir
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
        # garder le contre de sa grosse menace (Géant, Cochon…) tant qu'elle peut revenir dans sa main
        if set(self.opp_hand) & WIN_CONDITIONS:
            playable = {c: i for c, i in playable.items() if c not in ("cannon", "mini-pekka")}
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

    @staticmethod
    def _future(u: Seen, t: float) -> tuple[float, float]:
        """Position de l'unité dans t secondes (vitesse du suivi, bornée contre le bruit)."""
        vx, vy = u.vx / PHONE.tw, u.vy / PHONE.th          # cases/s
        v = math.hypot(vx, vy)
        k = min(1.0, MAX_UNIT_SPEED_TILES / v) if v > 0 else 0.0
        return u.x + u.vx * k * t, u.y + u.vy * k * t

    @staticmethod
    def _impact_time(card: str, x: float, y: float) -> float:
        return ACTION_DELAY_S + _tile_dist(x, y, 0.5, KING_Y) / SPELL_SPEED_TILES.get(card, 10.0)

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
        if "cannon" in playable and not is_air and (is_tank or t.name in ("hog-rider", "hog", "battle-ram", "royal-hog")):
            x = 0.5 + (-0.03 if lane_x < 0.5 else 0.03)
            return Decision("cannon", playable["cannon"], x, CANNON_Y, f"défense : {t.name} -> canon au centre, 4 cases sous la rivière")
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

    def _attack(self, seen: list[Seen], playable: dict, elixir: float, now: float) -> Decision | None:
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
        giant_at = self.p["giant_elixir"]
        # ses contres au Géant connus et tous hors de sa main (joués récemment) : fenêtre pour lancer plus tôt
        known = set(self.opp_deck) & GIANT_COUNTERS
        if self.p.get("giant_when_counter_out") and known and not known & set(self.opp_hand):
            giant_at -= 2
        if "giant" in playable and (elixir >= giant_at or punish):
            lane = self._weak_lane(seen)
            if self.p["counter_push"] and self.last_defense_lane is not None:
                lane = self.last_defense_lane      # contre-attaque avec les survivants de la défense
            self.giant_lane, self.giant_time = lane, now
            spot = self.p["giant_spot"]
            lx = LANES_X[lane] + ((-0.12 if lane == 0 else 0.12) if spot == "corner" else 0)
            x, y = _clamp_own(lx, {"back": 0.69, "corner": 0.69, "mid": 0.55, "bridge": 0.47}.get(spot, 0.69))
            why = "l'ennemi est à sec" if punish and elixir < giant_at else \
                "ses contres sont joués" if giant_at < self.p["giant_elixir"] and elixir < self.p["giant_elixir"] else \
                {"back": "au fond", "corner": "dans le coin", "mid": "au milieu", "bridge": "au pont"}.get(spot, spot)
            if punish and elixir < giant_at:
                x, y = _clamp_own(LANES_X[lane], 0.47)   # pression immédiate au pont
            return Decision("giant", playable["giant"], x, y, f"attaque : Géant ({why})")
        # soutien derrière notre Géant pendant qu'il avance
        ours = [s for s in seen if not s.enemy and s.name == "giant"]
        if ours and elixir >= self.p["support_min_elixir"]:
            g = ours[0]
            for card in ("musketeer", "archers", "valkyrie", "mini-pekka", "minions"):
                if card in playable:
                    x, y = _clamp_own(g.x, g.y + 0.07)
                    return Decision(card, playable[card], x, y, f"soutien : {card} derrière le Géant")
        if elixir >= self.p["cycle_at"] and playable:
            # élixir plein et rien à faire : on fait tourner la carte la moins chère, sans risque
            card = min((c for c in playable if DECK[c].kind == "troop"), key=lambda c: DECK[c].cost, default=None)
            if card == "archers":
                # au centre juste devant le Roi : les deux archères partent chacune dans un couloir,
                # une seule Flèche ne peut plus les prendre ensemble
                x, y = _clamp_own(0.5, OWN_TOWER_Y + 0.0)
                return Decision(card, playable[card], x, y, "rien à faire : archères séparées au centre")
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
