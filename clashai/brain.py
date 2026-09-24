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

# Géométrie de l'arène (mesurée sur l'écran du REDMAGIC, flux 578x1280)
RIVER_Y = 0.425
LANES_X = (0.205, 0.795)
OWN_ZONE = (0.04, 0.96, 0.46, 0.695)       # où l'on peut poser une troupe (la clôture du fond est à 0.705)
OWN_TOWER_Y, KING_Y = 0.59, 0.66
SPELL_LEAD_S = 0.6                          # on vise où sera l'unité quand le sort tombe


@dataclass
class Decision:
    card: str
    slot: int
    x: float
    y: float
    reason: str


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

    # ---- perception -> monde simplifié ----
    @staticmethod
    def to_seen(units: list[Unit], w: int, h: int, trails: dict | None, fps: float) -> list[Seen]:
        out = []
        for u in units:
            if u.name in SPAWNERS and u.enemy:
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
        playable = {c: i for i, c in enumerate(hand) if c in DECK and ready[i] and DECK[c].cost <= elixir + 0.3}
        enemies = [s for s in seen if s.enemy]
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
        if threats:
            d = self._defend(threats, playable)
            if d:
                self.last_defense_lane = _lane(d.x)
            return d
        return self._attack(seen, playable, elixir, now)

    def _spells(self, enemies: list[Seen], playable: dict) -> Decision | None:
        for card, min_count, kinds in (("arrows", self.p["arrows_min"], SWARM_UNITS),
                                       ("fireball", self.p["fireball_min"], None)):
            if card not in playable:
                continue
            pool = [e for e in enemies if kinds is None or e.name in kinds]
            if card == "fireball":
                pool = [e for e in pool if e.name not in SWARM_UNITS or e.name in ("barbarian", "archer")]
            r = DECK[card].radius
            best, center = 0, None
            for e in pool:
                group = [o for o in pool if math.hypot(o.x - e.x, (o.y - e.y) / 2.2) < r]
                if len(group) > best:
                    best = len(group)
                    center = (sum(o.x + o.vx * SPELL_LEAD_S for o in group) / len(group),
                              sum(o.y + o.vy * SPELL_LEAD_S for o in group) / len(group))
            if best >= min_count and (best >= 2 or card != "fireball" or
                                      any(e.name in TANK_UNITS or e.name in ("musketeer", "wizard", "witch",
                                          "executioner", "princess", "dart-goblin") for e in pool)):
                return Decision(card, playable[card], *center, f"{card} sur un groupe de {best}")
        return None

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
        for card in order:
            if card not in playable:
                continue
            c = DECK[card]
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
        if "giant" in playable and (elixir >= self.p["giant_elixir"] or punish):
            lane = self._weak_lane(seen)
            if self.p["counter_push"] and self.last_defense_lane is not None:
                lane = self.last_defense_lane      # contre-attaque avec les survivants de la défense
            self.giant_lane, self.giant_time = lane, now
            spot = self.p["giant_spot"]
            back = spot == "back"
            x, y = _clamp_own(LANES_X[lane], {"back": 0.69, "mid": 0.55, "bridge": 0.47}.get(spot, 0.69))
            why = "l'ennemi est à sec" if punish and elixir < self.p["giant_elixir"] else {"back": "au fond", "mid": "au milieu", "bridge": "au pont"}.get(spot, spot)
            if punish and elixir < self.p["giant_elixir"]:
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
