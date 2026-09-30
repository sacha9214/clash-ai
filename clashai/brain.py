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
from clashai import card_info
from clashai.tiles import OWN_FIRST_ROW, PHONE

# Géométrie de l'arène (mesurée sur l'écran du REDMAGIC, flux 578x1280)
RIVER_Y = 0.425
# grandes unités : le scanner les coupe souvent en 2 boîtes à 3-4 cases l'une de l'autre (30/09 : 2e boîte de notre
# Géant à 4,1 cases du point de pose -> restée « ennemie » -> Gargouilles sur notre propre Géant)
TALL_UNITS = {"giant", "golem", "pekka", "mega-knight", "giant-skeleton", "electro-giant", "goblin-giant", "royal-giant",
              "ice-golem", "elixir-golem-big", "lava-hound"}
SIEGE_WAIT_S = 8.0                           # entre deux troupes envoyées sur son Mortier / X-Bow
SIEGE_UNITS = {"mortar", "x-bow", "mortar-evolution", "x-bow-evolution"}
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


# Une de « nos » unités de l'autre côté de la rivière sans qu'on ait posé cette carte depuis ce délai : le détecteur
# s'est trompé de camp (match du 25/09 : Géant ennemi vu « à nous » -> « soutien derrière le Géant » fantôme)
OWN_UNIT_LIFE_S = 40.0
MUSKETEER_BONUS = 1.5                       # préférence pour la Mousquetaire en défense (voir _stat_pick)
RANGED_SAFE = 4.5                           # un tireur est posé à >= 4,5 cases de tout ennemi (après 1,5 s d'avance)
DEFENSE_WAIT_WIN_S = 7.0                    # … et 7 s si le 1er défenseur est donné gagnant
DEFENSE_WAIT_S = 4.0                        # après une défense : pas de 2e carte sur la même attaque avant 4 s
DEFENSE_RESERVE = 3                         # élixir toujours gardé pour défendre (Chevalier / Canon)
GIANT_FOLLOW_ELIXIR = 8                     # Géant (5) + de quoi le soutenir (3) : sinon on ne le lance pas
# unité lue -> (sa carte, unité qu'elle est probablement, carte de celle-ci) : confusions vues sur nos matchs
# unité lue -> (sa carte, [(unité qu'elle est probablement, carte de celle-ci), ...] essayées dans l'ordre) :
# confusions vues sur nos matchs. Remplacée seulement si sa carte n'est pas dans son deck connu et le candidat oui.
ALIAS = {"pekka": ("pekka", [("mini-pekka", "mini-pekka"), ("lumberjack", "lumberjack"), ("dark-prince", "dark-prince"),
                             ("knight", "knight"), ("barbarian", "barbarians"), ("valkyrie", "valkyrie")]),
         "mini-pekka-hero": ("mini-pekka", [("mini-pekka", "mini-pekka")]),
         "golden-knight": ("golden-knight", [("knight", "knight"), ("barbarian", "barbarians")]),
         "barbarian": ("barbarians", [("knight", "knight"), ("elite-barbarian", "elite-barbarians")]),
         "knight": ("knight", [("barbarian", "barbarians"), ("golden-knight", "golden-knight")])}
TOWER_DPS = 60.0                            # tour princesse, niveau 1 (même échelle que card_info.combat)
TOWER_RANGE_TILES = 7.5
# fin de match : 30 dernières secondes du temps réglementaire (3:00), puis prolongation = mort subite
LATE_S, REGULATION_S = 150.0, 180.0
PUSH_MEMORY_S = 8.0
DUPLICATE_S = 8.0                           # une « ennemie » collée à notre unité de même nom, posée il y a < 8 s : doublon
COUNTER_WINDOW_S = 12.0                     # contre-attaque : nos défenseurs posés il y a moins de 12 s
STURDY_HP = 450                             # PV (niveau 1) d'un défenseur qui vaut la peine d'être soutenu                         # défenses contre la même attaque adverse : créditée une fois
TANK_DEADLINE_S = 11.0                      # 12 s laissait les Gargouilles « gagner » contre un Géant (29/09) : il tape la tour avant
ONE_SHOT_HP = 55                            # PV (niveau 1) qu'un tir de tour princesse suffit à tuer (Squelette, Chauve-souris)
SPIRITS = {"fire-spirit", "ice-spirit", "electro-spirit", "heal-spirit"}                      # un tank qui vise les bâtiments, du pont à notre tour : ~9 s de marche + 1er coup
# Valeur d'un sort (élixir détruit) : une unité sur notre moitié frappe déjà nos tours/troupes -> compte plus ;
# une tour ennemie dans le cercle encaisse aussi (dégâts réduits aux tours)
OUR_SIDE_BONUS = 1.3
SPELL_TOWER_VALUE = {"fireball": 1.0, "arrows": 0.4}
# Achever une tour : dégâts du sort sur une tour / PV d'une tour princesse (niveau 11 : Boule de feu 207, Flèches 93,
# tour 3052 ; même rapport à tout niveau égal). Marge 0.8 : on ne tire que si le sort la détruit à coup sûr.
TOWER_HP_L11 = 3052
FINISH_MARGIN = 0.8

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
    return math.hypot(dx, dy) < max(SPELL_RADIUS_TILES.get(card, 0), card_info.spell_radius(card)) + 0.3   # marge d'imprécision


# ---- Adaptation au deck adverse (cartes vues par opponent.py) ----
FAST_BUILDING_HUNTERS = {"hog-rider", "hog", "battle-ram", "ram-rider", "royal-hog"}
SLOW_TANKS = {"giant", "golem", "giant-skeleton", "electro-giant", "goblin-giant", "elixir-golem-big"}
BIG_SPELLS = {"fireball", "poison", "lightning", "rocket", "earthquake"}
SMALL_SPELLS = {"arrows", "zap", "the-log", "giant-snowball", "barbarian-barrel", "tornado"}
PULL_BUILDINGS = {"cannon", "tesla", "inferno-tower", "bomb-tower", "goblin-cage", "tombstone"}


_TOWER_TILES = None


def _tower_tiles():
    """Cases occupées par nos tours (interdites au modèle de placement)."""
    global _TOWER_TILES
    if _TOWER_TILES is None:
        import numpy as np
        _TOWER_TILES = np.zeros((32, 18), bool)
        for r in range(32):
            for c in range(18):
                x, y = PHONE.center(c, r)
                _TOWER_TILES[r, c] = any(abs(x - cx) < hw and abs(y - cy) < hh for cx, cy, hw, hh in OWN_TOWERS)
    return _TOWER_TILES


def _cost(unit_name: str) -> int:
    v = UNIT2CARD.get(unit_name)
    return v[1] if v else 3


def _unit_value(unit_name: str) -> float:
    """Élixir que vaut UNE unité : coût de sa carte / nombre d'unités (un gobelin sur 3 = 0.67)."""
    v = UNIT2CARD.get(unit_name)
    if v:
        return v[1] / max(v[2], 1)
    st = card_info.combat(unit_name)
    return st["cost"] / max(st["count"], 1)


@dataclass
class Decision:
    card: str
    slot: int
    x: float
    y: float
    reason: str
    tile: tuple[int, int] | None = None   # case de la grille 18x32 où la carte est posée
    # placement tactique précis (Canon qui attire, pont, archères séparées…) : le modèle des pros ne le déplace pas
    # (Canon : 2.0 cases d'écart avec les pros pour la règle, 4.5 en moyenne pour le modèle)
    precise: bool = False
    # échange d'élixir estimé à la décision (défense, sort) : ce que l'on gagne (+) ou perd (-) ; None pour une attaque
    trade: float | None = None
    push: float = 0.0          # élixir de l'attaque adverse à laquelle cette défense répond (créditée une seule fois)


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
        self.last_defense_t = -1e9
        self.opp_elixir = 5.0        # estimation fournie par le modèle de l'adversaire
        self.opp_hand: list[str] = []            # sa main probable (cartes connues hors des 4 dernières)
        self.opp_deck: list[str] = []
        self.opp_heavy_t = -1e9                  # instant de sa dernière carte à 6+ élixir
        self._ours: list[Seen] = []
        self.match = None                        # towers.MatchState : PV de nos tours, tours ennemies, phase
        self.last_own_play: tuple[int, float] | None = None   # (couloir, instant) de notre dernière troupe
        self.virtual: list[dict] = []            # nos troupes posées, suivies de mémoire si le détecteur les rate
        # « où poser » appris sur les coups des pros (clashai/placement.py) ; les règles gardent « quand » et « quoi »
        self.placer = None
        if self.p.get("placement_model"):
            from clashai.placement import load
            self.placer = load()
        self.own_recent: list[tuple[tuple[str, ...], float, float, float]] = []   # (unités, x, y, instant) posées par nous
        self.own_lane: dict[str, int] = {}      # unité -> couloir de sa dernière pose
        self.last_defense_building = False
        self.last_defense: tuple[float, int, float] | None = None   # (instant, couloir, élixir d'attaque couvert)
        self.musk_plays = 0                     # poses de notre Mousquetaire : la 3e, 6e… est ÉVOLUÉE (Cycles 2)
        self.own_played: dict[str, float] = {}   # unité -> dernier instant où l'on a posé sa carte
        self.trades: list[tuple[float, float]] = []   # (instant, élixir gagné) de chaque échange vraiment joué
        self._credited: list[tuple[float, int]] = []    # (instant, couloir) des attaques adverses déjà créditées
        self._board = 0.0                              # élixir sur le terrain : nous - lui (mis à jour par decide)

    # ---- perception -> monde simplifié ----
    @staticmethod
    def to_seen(units: list[Unit], w: int, h: int, trails: dict | None, fps: float) -> list[Seen]:
        out = []
        for u in units:
            barrel = u.name in ("goblin-barrel", "goblin-barrel-evolution")     # Tonneau évolué : même défense
            if (u.name in SPAWNERS or barrel) and u.enemy:
                out.append(Seen("goblin-barrel" if barrel else u.name, True,
                                (u.box[0] + u.box[2]) / 2 / w, (u.box[1] + u.box[3]) / 2 / h))
                continue
            if u.name in BUILDINGS or u.name in NOT_UNITS:
                continue
            x, y = (u.box[0] + u.box[2]) / 2 / w, u.box[3] / h
            vx = vy = 0.0
            vel = getattr(u, "vel", None)
            if vel is not None:
                # vitesse du détecteur (px/s, horodatée) : juste même quand le détecteur tourne aussi pendant la pose
                # d'une carte (images rapprochées) — l'ancien calcul supposait 3 images = 3 / fps de la boucle
                vx, vy = vel[0] / w, vel[1] / h
            else:
                tr = trails.get(u.track_id) if trails else None
                if tr and len(tr) >= 4 and fps > 0:
                    (ax, ay), (bx, by) = tr[-4], tr[-1]
                    dt = 3 / fps
                    vx, vy = (bx - ax) / w / dt, (by - ay) / h / dt
            out.append(Seen(u.name, u.enemy, x, y, vx, vy))
        return out

    def _virtual_units(self, seen: list[Seen], now: float) -> list[Seen]:
        """Nos unités posées que le détecteur ne voit pas (ex. nos Archères, mal reconnues sur notre écran) :
        l'IA sait où elle les a posées ; elle les fait avancer vers l'ennemi à la vitesse réelle de la carte,
        s'arrêter quand un ennemi est à leur portée, et les oublie dès que le détecteur les voit (ou après 14 s).
        Elles encaissent les dégâts des ennemis et des tours à portée : une troupe morte ne « couvre » plus rien."""
        out, keep = [], []
        for v in self.virtual:
            if now > v["until"]:
                continue
            seen_real = any(not s.enemy and s.name == v["name"] and _tile_dist(s.x, s.y, v["x"], v["y"]) < 2.5
                            for s in seen)
            if seen_real and now - v["t"] > 1.2:
                continue                               # le détecteur la voit : la vraie détection prend le relais
            if now > v["last"]:
                dt, v["last"] = now - v["last"], now
                enemy_near = any(s.enemy and _tile_dist(s.x, s.y, v["x"], v["y"]) <= v["range"] + 1 for s in seen)
                if not enemy_near and v["y"] > ENEMY_TOWER_Y + 0.02:
                    v["y"] -= v["speed"] * dt          # avance vers la tour ennemie
                v["hp"] -= self._dps_on(v, seen) * dt
                if v["hp"] <= 0:
                    continue                           # tuée (estimation) : on l'oublie
            keep.append(v)
            if not seen_real:
                out.append(Seen(v["name"], False, v["x"], v["y"]))
        self.virtual = keep
        return out

    def _dps_on(self, v: dict, seen: list[Seen]) -> float:
        """Dégâts/s reçus par une de nos troupes virtuelles : ennemis qui peuvent la viser et l'ont à portée, tours."""
        dps = 0.0
        for s in seen:
            if not s.enemy:
                continue
            st = card_info.combat(s.name)
            if st["buildings_only"] or (v["flying"] and not st["hits_air"]):
                continue
            if _tile_dist(s.x, s.y, v["x"], v["y"]) <= st["range"] + 1:
                dps += st["dps"]
        alive = self.match.enemy_alive if self.match is not None else {0: True, 1: True}
        for lane, lx in enumerate(LANES_X):
            if alive[lane] and _tile_dist(lx, ENEMY_TOWER_Y, v["x"], v["y"]) <= TOWER_RANGE_TILES:
                dps += TOWER_DPS
        return dps

    def _fix_sides(self, seen: list[Seen], now: float) -> list[Seen]:
        """Corrige le camp deviné par le détecteur avec ce que l'IA sait de ses propres coups."""
        self.own_recent = [r for r in self.own_recent if now - r[3] < 6]
        # HÉROS : même rôle que la carte de base pour toutes nos règles (tank, Canon qui attire, contres…).
        # 30/09 : son Géant héros, lu « giant-hero », échappait aux règles du Géant.
        seen = [Seen(s.name[:-5], s.enemy, s.x, s.y, s.vx, s.vy) if s.name.endswith("-hero") else s for s in seen]
        own_units = {u for c in DECK.values() for u in c.units}
        unit_card = {u: c.name for c in DECK.values() for u in c.units}
        # même unité vue DEUX fois (à nous + ennemie) au même endroit : le double « ennemi » est un fantôme
        # (27/09 : notre Géant doublé d'un Géant ennemi -> Canon et Gargouilles contre notre propre Géant)
        # confusions fréquentes du scanner : un nom absent de son deck connu, dont le « voisin » y est -> le voisin
        # (27/09 : son Mini P.E.K.K.A lu « P.E.K.K.A » 10 fois -> menace surestimée)
        if len(self.opp_deck) >= 4:
            deck = set(self.opp_deck)
            def real(s):
                if not s.enemy or s.name not in ALIAS or ALIAS[s.name][0] in deck:
                    return s
                unit = next((u for u, c in ALIAS[s.name][1] if c in deck), None)
                return Seen(unit, s.enemy, s.x, s.y, s.vx, s.vy) if unit else s
            seen = [real(s) for s in seen]
        mine = [s for s in seen if not s.enemy]
        seen = [s for s in seen if not (s.enemy and s.name in own_units
                                        and any(m.name == s.name and _tile_dist(s.x, s.y, m.x, m.y) < 2.5 for m in mine))]
        out = []
        for s in seen:
            base = s.name.replace("-evolution", "")
            if s.enemy and s.y > RIVER_Y and any(s.name in names and _tile_dist(s.x, s.y, x, y) < (5 if s.name in TALL_UNITS else 4)
                                                 for names, x, y, _ in self.own_recent):
                # NOTRE unité fraîchement posée prise pour une ennemie (ex. notre Géant « ennemi » sur sa case de pose)
                s = Seen(s.name, False, s.x, s.y, s.vx, s.vy)
            elif s.enemy and (any(v["name"] == base and now < v["until"]
                                  # volantes : elles partent en diagonale vers un couloir, pas tout droit -> plus large
                                  and _tile_dist(s.x, s.y, v["x"], v["y"]) < (7 if base in AIR_UNITS else 4)
                                  for v in self.virtual)
                              or (base in own_units and s.y > RIVER_Y and s.vy < 0
                                  and now - self.own_played.get(base, -1e9) < OWN_UNIT_LIFE_S)):
                # une « ennemie » dans NOTRE moitié qui s'éloigne de nos tours, d'une carte qu'on vient de jouer :
                # c'est la nôtre qui part attaquer (27/09 : Canon + Gargouilles sur notre Géant, 13 s après la pose)
                # là où NOTRE troupe de même nom doit être (suivie de mémoire depuis sa pose, à sa vitesse) :
                # c'est la nôtre (27/09 : nos Gargouilles vues « ennemies » 4 s après la pose -> Chevalier gâché)
                s = Seen(s.name, False, s.x, s.y, s.vx, s.vy)
            elif (s.enemy and base in own_units and len(self.opp_deck) >= 4
                  and unit_card.get(base) not in self.opp_deck
                  and now - self.own_played.get(base, -1e9) < OWN_UNIT_LIFE_S
                  and self.own_lane.get(base) == _lane(s.x)):
                # (dans le couloir où on l'a posée : son 1er Chevalier dans l'autre couloir reste un ennemi)   # immobile en frappant une tour : sans condition de vitesse
                # « ennemie » d'une carte qu'il n'a pas, qu'on vient de jouer, et qui monte vers ses tours :
                # c'est la nôtre (27/09 : nos Gargouilles prises pour des ennemies -> Mousquetaire gâchée)
                s = Seen(s.name, False, s.x, s.y, s.vx, s.vy)
            elif not s.enemy and s.y < RIVER_Y - 0.02 and (
                    base not in own_units or now - self.own_played.get(base, -1e9) > OWN_UNIT_LIFE_S):
                # « à nous » chez l'ennemi alors qu'on n'a pas posé cette carte récemment : erreur de camp.
                # Elle descend vers nous -> ennemie ; sinon on l'ignore (ni alliée fantôme, ni cible de nos sorts)
                if s.vy / PHONE.th < 0.3:
                    continue
                s = Seen(s.name, True, s.x, s.y, s.vx, s.vy)
            out.append(s)
        # 2e passe anti-doublon, APRÈS les corrections de camp : la 1re tourne quand les deux boîtes sont encore
        # « ennemies ». Un grand Géant est souvent vu en 2 boîtes à ~3 cases d'écart (29/09 : la 2e restait ennemie
        # -> Mini P.E.K.K.A / Gargouilles sur notre propre Géant 2 s après sa pose)
        # seulement juste après NOTRE pose de cette carte : sinon un vrai ennemi de même nom (Chevalier contre Chevalier)
        # qui combat le nôtre disparaîtrait des menaces
        mine = [s for s in out if not s.enemy]
        out = [s for s in out if not (s.enemy and s.name in own_units
                                      and now - self.own_played.get(s.name, -1e9) < DUPLICATE_S
                                      and any(m.name == s.name and _tile_dist(s.x, s.y, m.x, m.y) < 3.5 for m in mine))]
        return out

    # ---- décision ----
    def decide(self, seen: list[Seen], hand: list[str | None], ready: list[bool], elixir: float,
               now: float) -> Decision | None:
        """Choisit un coup SANS rien retenir : la boucle appelle played() seulement si la carte est vraiment posée
        (sinon, pendant le délai entre deux cartes, chaque décision non jouée créait des troupes virtuelles fantômes)."""
        seen = self._fix_sides(seen, now)
        seen = seen + self._virtual_units(seen, now)
        self.last_seen = seen                   # ce que le cerveau voit après ses corrections (journal : diagnostic)
        self._ours = [s for s in seen if not s.enemy]
        # élixir déjà engagé sur le terrain (un Géant qui avance compte encore 5 pour nous)
        self._board = sum(_unit_value(s.name) * (-1 if s.enemy else 1) for s in seen)
        d = self._decide(seen, hand, ready, elixir, now)
        if d is None:
            return None
        if d.card == "cannon" and not d.precise:
            # le Canon doit ATTIRER : toujours au centre, près de la rivière (jamais au fond, jamais sur l'ennemi)
            t = max((s for s in seen if s.enemy), key=lambda s: s.y, default=None)
            col = 8 if (t.x if t else d.x) < 0.5 else 9
            d.x, d.y = PHONE.center(col, OWN_FIRST_ROW + 3)
            d.precise = True
        if self.placer and DECK[d.card].kind != "spell" and self.placer.knows(d.card) and not d.precise:
            units = [(u.name, u.enemy, *PHONE.to_tile(u.x, u.y)) for u in seen]
            # couloir choisi par les règles (côté de l'ennemi : ~80 % d'accord avec les pros), case exacte par le
            # modèle ; une carte posée au centre par la règle laisse le modèle libre
            rule_lane = _lane(d.x) if abs(d.x - 0.5) > 0.06 else None
            (c, r), _ = self.placer.predict(d.card, units, forbid=_tower_tiles(), lane=rule_lane)
            d.x, d.y = PHONE.center(c, r)
            # la règle a choisi QUOI et QUAND ; l'endroit vient du modèle : on ne garde que la première partie
            d.reason = d.reason.split(" -> ")[0].split(" (")[0] + f" -> {d.card}, case apprise des pros ({c},{r})"
        if DECK[d.card].kind != "spell":
            # le jeu pose au centre d'une case : on vise ce centre (et on reste hors des tours)
            d.x, d.y = PHONE.snap(*_clamp_own(*PHONE.snap(d.x, d.y)))
        d.tile = PHONE.cell(d.x, d.y)
        return d

    def played(self, d: Decision, now: float) -> None:
        """La carte de `d` est vraiment posée (tap vérifié) : l'IA s'en souvient (camp de ses unités, troupes
        virtuelles, couloir de sa dernière troupe)."""
        card = DECK[d.card]
        if d.reason.startswith("défense") and card.kind != "spell":
            # un défenseur donné GAGNANT par les stats a besoin de temps (Mini P.E.K.K.A sur un Géant : ~10 s) :
            # on attend plus longtemps avant d'en rajouter (29/09 : Valkyrie ajoutée 4,1 s après)
            wait = DEFENSE_WAIT_WIN_S if "gagne en" in d.reason else DEFENSE_WAIT_S
            self.last_defense_building = card.kind == "building"
            self.last_defense = (now + wait - DEFENSE_WAIT_S, _lane(d.x), DECK[d.card].cost + 2)
        if d.card == "musketeer":
            self.musk_plays += 1
        if d.trade is not None:
            trade = d.trade
            if d.push:
                lane = _lane(d.x)
                # 2e défense contre la même attaque (même couloir, < 8 s) : son élixir est déjà compté, on ne compte
                # que ce que l'on dépense en plus (sinon chaque carte posée contre un Géant « gagnait » 5)
                if any(now - t < PUSH_MEMORY_S and l == lane for t, l in self._credited):
                    trade -= d.push
                self._credited = [(t, l) for t, l in self._credited if now - t < PUSH_MEMORY_S] + [(now, lane)]
            self.trades.append((now, round(trade, 2)))
        for u in card.units:
            self.own_played[u] = now
            self.own_lane[u] = _lane(d.x)
        if card.kind != "spell":
            self.own_recent.append((card.units, d.x, d.y, now))
            if abs(d.x - 0.5) > 0.06:                                  # pas une carte posée au centre
                self.last_own_play = (_lane(d.x), now)
        if card.kind == "troop":
            st = card_info.combat(d.card)
            name = card.units[0] if card.units else d.card
            for i in range(min(st["count"], 3)):
                off = (i - (min(st["count"], 3) - 1) / 2) * 0.6 * PHONE.tw
                self.virtual.append({"name": name, "x": d.x + off, "y": d.y, "t": now, "last": now + 1.0,   # déploiement ~1 s
                                     "speed": (card_info.info(d.card)["speed"] or 1.0) * PHONE.th,
                                     "range": st["range"], "until": now + 14, "hp": st["hp"] or 300.0,
                                     "flying": bool(st["flying"])})
        if d.card == "giant":
            self.giant_lane, self.giant_time = _lane(d.x), now
        if d.reason.startswith("punition"):
            self.opp_heavy_t = -1e9                                    # une seule punition par carte lourde
        if d.reason.startswith("défense"):
            self.last_defense_lane, self.last_defense_t = _lane(d.x), now

    def _decide(self, seen: list[Seen], hand: list[str | None], ready: list[bool], elixir: float,
                now: float) -> Decision | None:
        playable = {c: i for i, c in enumerate(hand) if c in DECK and ready[i] and DECK[c].cost <= elixir + 0.3}
        enemies = [s for s in seen if s.enemy]
        # SIÈGE : son Mortier / X-Bow tire sur notre tour DEPUIS SA MOITIÉ -> ce n'est jamais une « menace » au sens des
        # défenses, personne n'y allait (30/09 : Mortier intact toute la partie, Mini P.E.K.K.A jamais joué). On envoie
        # un tueur au sol au pont de ce couloir.
        siege = [e for e in enemies if e.name in SIEGE_UNITS and e.y < RIVER_Y]
        if siege and now - getattr(self, "last_siege", -1e9) > SIEGE_WAIT_S:   # le 1er tueur a le temps d'arriver
            m = max(siege, key=lambda e: e.y)
            for card in ("mini-pekka", "knight", "valkyrie", "giant"):
                if card in playable:
                    self.last_siege = now
                    x, y = _clamp_own(LANES_X[_lane(m.x)], RIVER_Y + 0.03)
                    return Decision(card, playable[card], x, y, f"siège : {m.name} -> {card} au pont pour le détruire",
                                    precise=True)
        # Tonneau à gobelins en vol : Valkyrie juste derrière la tour visée, elle balaie les 3 gobelins à l'atterrissage
        barrel = next((e for e in enemies if e.name == "goblin-barrel"), None)
        if barrel and len(self.opp_deck) >= 8 and "goblin-barrel" not in self.opp_deck:
            # deck complet, sans Tonneau : fausse détection (27/09 : 2 Valkyries gâchées). Pas dès 6 cartes : un sort
            # n'entre dans son deck qu'à la 2e fois, son 1er vrai Tonneau aurait toujours été ignoré
            barrel = None
        if barrel and "valkyrie" in playable:
            lane_x = LANES_X[_lane(barrel.x)]
            x, y = _clamp_own(lane_x, OWN_TOWER_Y + 0.06)
            gobs = [Seen("goblin", True, barrel.x, barrel.y)] * 3        # ce qui va atterrir
            return Decision("valkyrie", playable["valkyrie"], x, y, "défense : Tonneau à gobelins -> Valkyrie derrière la tour",
                            precise=True, trade=self._trade("valkyrie", gobs, True, _cost("goblin-barrel")),
                            push=_cost("goblin-barrel"))
        enemies = [e for e in enemies if e.name != "goblin-barrel"]
        threats = [s for s in enemies if s.y > RIVER_Y - self.p['defend_line']   # sur notre moitié ou au pont
                   and s.name not in SPAWNERS       # ses bâtiments ne viennent pas : ce qui en sort sera défendu
                   # dans SA moitié et qui s'éloigne de nous : aucune menace (29/09 : notre Géant lu « ennemi » juste
                   # après le pont -> Canon + Chevalier contre notre propre Géant)
                   and not (s.y < RIVER_Y and s.vy < -0.3 * PHONE.th)]

        d = self._finish_tower(playable)
        if d:
            return d
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
        if threats and self.p.get("stat_defense", True) and not any(self._tower_low(_lane(t.x)) for t in threats):
            # dégâts que la tour prendrait si on ne fait RIEN : leurs dégâts/s x le temps que la tour met à les tuer
            # seule (~60 dégâts/s, niveau 1). Petit (< ~12 % d'une tour) : la tour s'en charge, on garde l'élixir
            # (match du 27/09 : Chevalier puis Mini P.E.K.K.A sur un Golem de glace, Canon sur 1 squelette)
            st = [card_info.combat(t.name) for t in threats]
            hp, dps = sum(s["hp"] for s in st), sum(s["dps"] for s in st)
            # stats inconnues (Boss Bandit, Petit Prince… : PV ou dégâts à 0 dans la base) : « inoffensif » serait
            # faux -> on défend
            if all(s["hp"] and s["dps"] for s in st) and dps * hp / 60 < 170:
                threats = []
        if threats and not any(self._tower_low(_lane(t.x)) for t in threats) and len(threats) <= 4 and all(
                t.name in SPIRITS or 0 < card_info.combat(t.name)["hp"] <= ONE_SHOT_HP for t in threats):
            # 2-4 Squelettes / esprits : la tour les tue en 1 coup (les esprits meurent en frappant) (27/09 : 5 cartes
            # gâchées). Pas les Gargouilles ni les Gobelins (79-90 PV : plusieurs coups, 3 Gargouilles = 138 dégâts/s)
            threats = []
        if threats:
            # une nuée au sol (3+ unités) et notre Valkyrie en main, presque payable : on l'attend (~1 s) plutôt
            # que de jeter un Chevalier qui « ralentit seulement » (27/09 : 5 Barbares, Chevalier à 2,8 élixirs)
            t0 = max(threats, key=lambda s: s.y)
            crowd = [s for s in threats if s.name not in AIR_UNITS and _tile_dist(s.x, s.y, t0.x, t0.y) < 3]
            vi = next((i for i, c in enumerate(hand) if c == "valkyrie" and ready[i]), None)
            if (len(crowd) >= 3 and vi is not None and "valkyrie" not in playable
                    and elixir >= DECK["valkyrie"].cost - 1.5 and t0.y < OWN_TOWER_Y - 0.06 and not self._tower_low(_lane(t0.x))):
                return None
            # SURDÉFENSE : une défense vient d'être posée dans ce couloir contre une attaque modeste -> on laisse
            # le défenseur faire son travail avant d'en rajouter (29/09 : Mini P.E.K.K.A + Canon + Chevalier = 11 élixirs
            # en 4 s contre un Bûcheron à 4)
            lane0 = _lane(t0.x)
            attack = sum(_cost(n) for n in {s.name for s in threats if _lane(s.x) == lane0})
            if (self.last_defense and now - self.last_defense[0] < DEFENSE_WAIT_S and self.last_defense[1] == lane0
                    and attack <= self.last_defense[2] + 2 and not self._tower_low(lane0)
                    and (self.last_defense_building
                         or any(not o.enemy and _lane(o.x) == lane0 for o in seen))):   # notre défenseur est encore là
                # (un bâtiment n'est jamais dans « seen » : on le compte présent pendant l'attente)
                return None
            return self._defend(threats, playable)
        # Canon posé à l'avance : son tank ou son Cochon descend vers le pont (38 % des Canons des pros)
        coming = [e for e in enemies if (e.name in TANK_UNITS or e.name in FAST_BUILDING_HUNTERS) and e.name not in AIR_UNITS
                  and RIVER_Y - 6 * PHONE.th < e.y <= RIVER_Y and e.vy >= 0]
        if coming and "cannon" in playable:
            t = max(coming, key=lambda e: e.y)
            row = OWN_FIRST_ROW + (4 if t.name in FAST_BUILDING_HUNTERS else 3)   # assez près pour l'attirer
            x, y = PHONE.center(8 if t.x < 0.5 else 9, row)
            return Decision("cannon", playable["cannon"], x, y, f"canon à l'avance : {t.name} arrive vers le pont",
                            precise=True)
        # garder le contre de sa grosse menace (Géant, Cochon…) tant qu'elle peut revenir dans sa main
        keep = set()
        if set(self.opp_hand) & WIN_CONDITIONS:
            keep |= {"cannon", "mini-pekka"}
        if "goblin-barrel" in self.opp_hand:
            keep.add("valkyrie")            # le seul contre propre au Tonneau
        d = self._attack(seen, {c: i for c, i in playable.items() if c not in keep}, elixir, now)
        if d is None and keep & set(playable) and elixir >= self._cycle_at(now):
            # élixir plein et seuls les contres gardés sont jouables : perdre de l'élixir coûte plus cher que de
            # jouer le contre maintenant (il reviendra en main ; l'élixir perdu, jamais)
            d = self._attack(seen, playable, elixir, now)
        return d

    def _counter_push(self, seen: list[Seen], playable: dict, elixir: float, now: float, support_at: float,
                      mode: str | None, edge: int) -> Decision | None:
        """Contre-attaque : un de nos défenseurs solides a survécu et repart vers le pont -> une carte derrière lui.
        L'élixir de la défense est déjà « rentabilisé » : ajouter du soutien est l'attaque la moins chère du jeu."""
        if (not self.p.get("counter_support") or mode == "défendre l'avance" or edge < 0 or elixir < support_at
                or now - self.last_defense_t > COUNTER_WINDOW_S):
            return None
        survivors = [s for s in seen if not s.enemy and s.name != "giant" and _lane(s.x) == self.last_defense_lane
                     and card_info.combat(s.name)["hp"] >= STURDY_HP and s.vy / PHONE.th < -0.2
                     and RIVER_Y < s.y < OWN_TOWER_Y + 0.02]
        if not survivors or any(e.enemy and e.y > RIVER_Y for e in seen):
            return None                                   # rien à pousser, ou encore un ennemi chez nous
        lead = min(survivors, key=lambda s: s.y)          # le plus avancé
        order = ["archers", "musketeer", "valkyrie", "mini-pekka", "knight"]
        if set(self.opp_hand) & SMALL_SPELLS:
            order = [c for c in order if c != "archers"] + ["archers"]
        reserve = 0 if self.opp_elixir < 3 or mode in ("tout pour l'attaque", "mort subite") else DEFENSE_RESERVE
        for card in order:
            if card in playable and elixir - DECK[card].cost >= reserve:   # toujours de quoi défendre derrière
                x, y = _clamp_own(lead.x, lead.y + 0.05)  # juste derrière lui : il encaisse, notre soutien tire
                return Decision(card, playable[card], x, y, f"contre-attaque : {card} derrière notre {lead.name}")
        return None

    def _edge(self, elixir: float) -> tuple[int, float]:
        """(+1 en avance / -1 en retard / 0, écart) : notre élixir + nos troupes sur le terrain, moins son élixir estimé
        (opponent.py) + ses troupes, comparé au seuil edge_push (0 = on ignore l'écart). Sans le terrain, l'IA se
        croyait « en retard » juste après avoir posé son Géant et ne le soutenait plus."""
        gap = elixir - self.opp_elixir + self._board
        k = self.p.get("edge_push", 0)
        return (0 if not k else 1 if gap >= k else -1 if gap <= -k else 0), gap

    def _endgame(self, now: float) -> str | None:
        """Fin de match selon les couronnes (towers.MatchState) : en tête dans les 30 dernières secondes -> on défend ;
        mené -> tout pour l'attaque ; prolongation (couronnes égales, la 1re tour qui tombe gagne) -> mort subite."""
        if not self.p.get("endgame") or self.match is None:
            return None
        t = self.match.elapsed(now)
        if t < LATE_S:
            return None
        ours, theirs = self.match.crowns(now)
        if t >= REGULATION_S and ours == theirs:
            return "mort subite"
        if ours > theirs:
            return "défendre l'avance"
        if ours < theirs:
            return "tout pour l'attaque"
        return None

    @property
    def trade_balance(self) -> float:
        """Bilan des échanges d'élixir du match (estimé à chaque défense / sort joué)."""
        return round(sum(t for _, t in self.trades), 1)

    def _cycle_at(self, now: float) -> float:
        """Élixir à partir duquel on joue plutôt que de perdre de l'élixir. La barre se lit au plus ~9.5 :
        un seuil à 10 n'était jamais atteint (l'IA restait pleine sans rien faire)."""
        fast = self.match is not None and self.match.phase(now) != "normal"
        # jamais sous le seuil du Géant : sinon on « fait tourner » une carte à 7 et le Géant (8) ne sort jamais
        # (29/09 : un seul Géant de la partie)
        return max(min(self.p["cycle_at"], 9.4) - (1.5 if fast else 0), GIANT_FOLLOW_ELIXIR + 0.5 - (1 if fast else 0))

    def _finish_tower(self, playable: dict) -> Decision | None:
        """Une tour ennemie presque morte : le sort qui l'achève à coup sûr = une couronne (si la lecture de ses PV a
        fait ses preuves pendant ce match, towers.MatchState.enemy_hp_ok)."""
        m = self.match
        if m is None or not getattr(m, "enemy_hp_ok", False) or not self.p.get("finish_towers", True):
            return None
        for card in ("arrows", "fireball"):                   # le moins cher qui suffit
            if card not in playable:
                continue
            share = card_info.tower_damage(card) / TOWER_HP_L11
            for lane in sorted((0, 1), key=lambda l: m.enemy_hp[l]):
                if m.enemy_alive[lane] and 0 < m.enemy_hp[lane] <= FINISH_MARGIN * share:
                    # centre sur la tour, ou décalé vers le bord du terrain (Flèches : rayon 3.5, le Roi est tout près)
                    spots = [(LANES_X[lane] + (-dx if lane == 0 else dx) * PHONE.tw, ENEMY_TOWER_Y) for dx in (0, 0.75, 1.5)]
                    spot = next((p for p in spots if not _hits_enemy_king(card, *p)), None)
                    if spot is None:
                        continue
                    x, y = spot
                    return Decision(card, playable[card], x, y,
                                    f"{card} achève la tour {'gauche' if lane == 0 else 'droite'} "
                                    f"({m.enemy_hp[lane]:.0%} de PV)", precise=True)
        return None

    def _spell_value(self, card: str, hit: list[tuple[Seen, tuple[float, float]]], center: tuple[float, float]) -> float:
        """Élixir détruit par le sort : pour chaque unité touchée, ce qu'elle vaut x (part de ses PV enlevée)².
        Une unité tuée compte entièrement ; des dégâts partiels comptent peu (elle continue de se battre).
        Ex. Boule de feu sur un Géant seul : ~0.1 élixir pour 4 (25/09 : 57 Boules de feu gâchées sur des tanks seuls)."""
        dmg = card_info.spell_damage(card)
        value = 0.0
        for o, (_, fy) in hit:
            hp = card_info.combat(o.name)["hp"]
            part = min(1.0, dmg / hp) if hp else 0.5
            value += _unit_value(o.name) * part ** 2 * (OUR_SIDE_BONUS if fy > RIVER_Y else 1.0)
        alive = self.match.enemy_alive if self.match is not None else {0: True, 1: True}
        r = card_info.spell_radius(card)
        if any(alive[lane] and _tile_dist(center[0], center[1], lx, ENEMY_TOWER_Y) < r + TOWER_HALF_TILES
               for lane, lx in enumerate(LANES_X)):
            value += SPELL_TOWER_VALUE.get(card, 0.0)
        return value

    def _spells(self, enemies: list[Seen], playable: dict) -> Decision | None:
        """Sort sur le groupe qui rapporte le plus d'élixir, s'il en rapporte au moins spell_value x son coût."""
        for card, min_count in (("arrows", self.p["arrows_min"]), ("fireball", self.p["fireball_min"])):
            if card not in playable or not enemies:
                continue
            r = card_info.spell_radius(card)                  # rayon réel, en cases
            t = self._impact_time(card, 0, 0)
            fut = [self._future(e, t) for e in enemies]        # où sera chaque unité quand le sort tombera
            best = None                                        # (valeur, nombre touché, centre)
            for sx, sy in fut:
                group = [p for p in fut if _tile_dist(*p, sx, sy) < r]
                c = (sum(p[0] for p in group) / len(group), sum(p[1] for p in group) / len(group))
                # on ne compte que les unités bien DEDANS au centre visé (marge de 15 % pour l'imprécision)
                hit = [(e, p) for e, p in zip(enemies, fut) if _tile_dist(*p, *c) < 0.85 * r]
                if not hit or _hits_enemy_king(card, *c):
                    continue   # groupe collé au Roi ennemi : on ne l'active pas pour quelques troupes
                v = self._spell_value(card, hit, c)
                if best is None or (v, len(hit)) > best[:2]:
                    best = (v, len(hit), c)
            if best is None:
                continue
            value, n, center = best
            if card == "fireball" and n < 2 and self.p.get("fireball_patient") and not any(
                    _tile_dist(center[0], center[1], lx, ENEMY_TOWER_Y) < r for lx in LANES_X):
                continue   # cible seule loin d'une tour : on attend qu'elle s'en approche ou qu'une 2e la rejoigne
            if n >= min_count and value >= self.p.get("spell_value", 0.8) * DECK[card].cost:
                return Decision(card, playable[card], *center,
                                f"{card} sur un groupe de {n} ({value:.1f} élixir détruits)", precise=True,
                                trade=round(value - DECK[card].cost, 2))
        return None

    def _future(self, u: Seen, t: float) -> tuple[float, float]:
        """Position de l'unité dans t secondes : elle avance (vitesse du suivi, bornée contre le bruit)
        jusqu'à ce qu'une de nos troupes ou tours soit à sa portée, puis elle s'arrête pour se battre."""
        vx, vy = u.vx / PHONE.tw, u.vy / PHONE.th          # cases/s
        v = math.hypot(vx, vy)
        if v <= 0 or t <= 0:
            return u.x, u.y
        k = min(1.0, MAX_UNIT_SPEED_TILES / v)
        st = card_info.info(u.name)                               # vraies statistiques (base des cartes)
        rng = st["range"] or ATTACK_RANGE_TILES.get(u.name, MELEE_RANGE_TILES)
        melee = rng <= MELEE_RANGE_TILES + 0.1
        blockers = [(bx, by, TOWER_HALF_TILES) for bx, by in OWN_TOWER_CENTERS]
        if not (st["buildings_only"] or u.name in BUILDING_TARGETERS):
            blockers += [(o.x, o.y, 0.5) for o in self._ours
                         if st["hits_air"] or not (o.name in AIR_UNITS or card_info.info(o.name)["flying"])]
        x, y, step = u.x, u.y, 0.1
        for _ in range(int(t / step) + 1):
            if any(_tile_dist(x, y, bx, by) <= rng + size for bx, by, size in blockers):
                break                                               # une cible à portée : elle s'arrête là
            x, y = x + u.vx * k * step, y + u.vy * k * step
        return x, y

    @staticmethod
    def _impact_time(card: str, x: float, y: float) -> float:
        """Temps entre la décision et l'impact. Boule de feu : elle part de notre Roi, le vol dépend de la distance
        (mesuré sur 258 tirs le 29/09 : 1,04 s + 0,047 s/case jusqu'à la 1re détection, +0,3 s jusqu'à l'explosion).
        L'ancien temps fixe de 2,0 s visait trop tard les cibles proches et trop tôt les lointaines."""
        if card == "fireball":
            return 1.35 + 0.047 * _tile_dist(x, y, 0.5, KING_Y)
        return SPELL_IMPACT_S.get(card, 2.0)

    def _ranged_spot(self, rng: float, group: list[Seen]) -> tuple[float, float, float | None]:
        """Meilleure case de notre moitié pour un tireur : chaque ennemi est pris là où il sera dans 1,5 s (apparition
        + premier tir). Case valable si l'ennemi le plus proche est à >= RANGED_SAFE cases (et hors de sa portée + 1),
        et une cible à portée. Parmi elles : la plus proche de notre tour (elle la protège), puis la plus sûre."""
        future = [(self._future(g, 1.5), card_info.combat(g.name)["range"]) for g in group]
        best, best_key = None, None
        for row in range(OWN_FIRST_ROW, 31):
            for col in range(0, 18):
                x, y = _clamp_own(*PHONE.center(col, row))
                ds = [(_tile_dist(x, y, fx, fy), r) for (fx, fy), r in future]
                near = min(d for d, _ in ds)
                if near < RANGED_SAFE or any(d < r + 1.0 for d, r in ds) or near > rng:
                    continue
                tower = min(_tile_dist(x, y, tx, ty) for tx, ty in OWN_TOWER_CENTERS)
                key = (round(tower), -near)
                if best_key is None or key < best_key:
                    best, best_key = (x, y, near), key
        if best is None:
            return 0.0, 0.0, None
        return best

    def evo_musketeer(self) -> bool:
        """La prochaine Mousquetaire posée est évoluée : 2 poses normales chargent l'évolution, la 3e est évoluée."""
        return self.musk_plays % 3 == 2

    @staticmethod
    def _duel(card: str, group: list[Seen], near_tower: bool) -> tuple[bool, float, float] | None:
        """Combat estimé entre notre carte et le groupe ennemi : (gagne ?, marge en s, temps pour tout tuer).
        Temps pour tuer = PV ennemis / nos dégâts par seconde ; temps pour mourir = nos PV / leurs dégâts par
        seconde (seulement les unités qui visent les troupes et peuvent nous toucher). Zone : touche jusqu'à 3."""
        me = card_info.combat(card)
        foes = [card_info.combat(g.name) for g in group]
        hittable = [f for f in foes if (f["flying"] and me["hits_air"]) or (not f["flying"] and me["hits_ground"])]
        if not hittable:
            return None                                     # ne peut rien toucher (ex. Chevalier contre un volant)
        # coup par coup : une unité mono-cible tue UNE cible à la fois (des squelettes à 1 coup = 1 coup chacun) ;
        # une unité à zone frappe jusqu'à 3 cibles à chaque coup. Nos unités se partagent les cibles.
        hits = sum(math.ceil(f["hp"] / me["dmg"]) if me["dmg"] else 99 for f in hittable)
        if me["splash"]:
            hits = math.ceil(hits / min(len(hittable), 3))
        t_kill = hits * me["hs"] / me["count"]
        if near_tower:                                       # la tour princesse tire aussi (~60 dégâts/s, niveau 1)
            t_kill = 1 / (1 / max(t_kill, 1e-6) + 60 / max(sum(f["hp"] for f in hittable), 1))
        their_dps = Brain._their_dps(card, foes)
        t_die = me["hp"] * me["count"] / their_dps if their_dps else 1e9
        if card == "witch":
            # ses squelettes (4 par vague) : chacun encaisse un coup à sa place et frappe aussi (30/09 : sans eux la
            # Sorcière ne « gagnait » jamais et n'était jamais jouée)
            t_die += 4 * max((f["hs"] for f in hittable), default=1.0)
            t_kill *= 0.75
        if any(f["buildings_only"] for f in foes):
            # un Géant ne nous frappe pas : il frappe la TOUR. « Gagner », c'est le tuer avant qu'il ne l'abîme
            # (sinon n'importe quelle carte « gagne » et la moins chère passait : Archères 17 s au lieu du Mini P.E.K.K.A 10 s)
            t_die = min(t_die, TANK_DEADLINE_S)
        return t_kill < t_die, t_die - t_kill, t_kill

    @staticmethod
    def _their_dps(card: str, foes: list[dict]) -> float:
        """Dégâts/s que le groupe ennemi inflige à notre carte (un Géant ne frappe que nos bâtiments, ex. le Canon)."""
        me = card_info.combat(card)
        building = card in DECK and DECK[card].kind == "building"
        return sum(f["dps"] * (min(me["count"], 3) if f["splash"] else 1) for f in foes
                   if (building or not f["buildings_only"]) and (not me["flying"] or f["hits_air"]))

    def _trade(self, card: str, group: list[Seen], near_tower: bool, push_cost: int) -> float:
        """Échange estimé d'une défense : élixir de l'attaque - ce que notre carte y laisse. Une troupe qui gagne et
        survit n'est pas « dépensée » (Mini P.E.K.K.A qui tue un Géant sans une égratignure : +5, pas +1)."""
        r = self._duel(card, group, near_tower)
        keep = 0.0
        if r and r[0]:
            me = card_info.combat(card)
            their = self._their_dps(card, [card_info.combat(g.name) for g in group])
            keep = max(0.0, 1 - r[2] * their / max(me["hp"] * me["count"], 1)) if their else 1.0
        return round(push_cost - DECK[card].cost * (1 - keep), 2)

    def _stat_pick(self, t: Seen, threats: list[Seen], playable: dict, push_cost: int) -> Decision | None:
        """Choisit la carte qui GAGNE le combat au moindre coût, puis la place hors de portée si c'est un tireur."""
        group = [s for s in threats if _tile_dist(s.x, s.y, t.x, t.y) < 4]
        near_tower = t.y > OWN_TOWER_Y - 0.1
        options = []
        for card in playable:
            if card not in DECK or DECK[card].kind == "spell" or card == "giant":
                continue
            if card == "cannon" and (len(group) >= 3 or card_info.combat(t.name)["count"] >= 3):
                continue                # le Canon tire sur 1 cible à la fois : inutile contre une nuée (29/09 : Barbares)
            if t.name in AIR_UNITS and not card_info.combat(card)["hits_air"]:
                continue                # ne peut pas les toucher (Mini P.E.K.K.A contre des Gargouilles) : élixir perdu
            r = self._duel(card, group, near_tower)
            if r:
                win, margin, t_kill = r
                # gagnants : le moins cher d'abord ; si personne ne gagne : celui qui tient le plus longtemps
                # la Mousquetaire compte 1 élixir de moins : elle tire de loin, survit souvent et repart en
                # contre-attaque, et chaque pose charge son évolution (29/09 : 1 Mousquetaire par partie)
                eff = DECK[card].cost - (MUSKETEER_BONUS if card in ("musketeer", "witch") else 0)   # tireurs qui survivent
                options.append((not win, eff > push_cost + 1, eff if win else 0, -margin,
                                eff, card, t_kill))
        if not options:
            return None
        # garder Canon / Mini P.E.K.K.A pour son Géant/Cochon s'il l'a en main et qu'une autre carte gagne ici
        if t.name not in WIN_CONDITIONS and set(self.opp_hand) & WIN_CONDITIONS:
            spare = [o for o in options if o[5] not in ("cannon", "mini-pekka") and not o[0]]
            if spare:
                options = spare
        options.sort()
        if self.evo_musketeer() and "musketeer" in playable and len(group) >= 2 and t.name not in AIR_UNITS | {"balloon"}:
            # Mousquetaire ÉVOLUÉE : 3 tirs de sniper (portée 6-30 cases, 2 cases de large) qui TRAVERSENT tout ce qui
            # est aligné -> la meilleure réponse à un groupe qui descend un couloir (tank + soutien)
            options = [o for o in options if o[5] == "musketeer"] or options
        lose, pricey, _, neg_margin, cost, card, t_kill = options[0]
        if card == "musketeer" and self.evo_musketeer():
            # en retrait derrière notre tour, dans l'axe du couloir : ses tirs remontent le couloir et percent la file
            lane_x = LANES_X[_lane(t.x)]
            x, y = _clamp_own(lane_x, OWN_TOWER_Y + 0.06)
            return Decision(card, playable[card], x, y,
                            f"défense : {t.name} x{len(group)} -> Mousquetaire évoluée derrière la tour (sniper le long du couloir)",
                            precise=True, trade=self._trade(card, group, True, push_cost), push=push_cost)
        if lose and pricey:
            return None                                     # rien ne gagne à bon prix : la tour encaisse
        if lose and not near_tower and t.name not in TANK_UNITS:
            # personne ne gagne seul au pont : on attend qu'ils entrent dans la portée de notre tour, puis on défend
            # à côté d'elle (tour + unité ensemble) — au lieu de perdre l'unité et l'élixir loin de la tour
            return None
        if card == "cannon":
            col = 8 if t.x < 0.5 else 9                        # au centre, près de la rivière : il attire l'ennemi
            x, y = PHONE.center(col, OWN_FIRST_ROW + (4 if t.name in FAST_BUILDING_HUNTERS else 3))
            return Decision(card, playable[card], x, y, f"défense : {t.name} x{len(group)} -> canon au centre (attire)",
                            precise=True, trade=self._trade(card, group, near_tower, push_cost), push=push_cost)
        me = card_info.combat(card)
        their_range = max(card_info.combat(g.name)["range"] for g in group)
        if me["range"] >= 4:
            # TIREUR (Mousquetaire, Sorcière) : jamais au contact. Case la plus sûre de notre moitié : à sa portée,
            # hors de portée de TOUS les ennemis (avance pendant l'apparition comprise), près de notre tour.
            # 30/09 : la moitié des Mousquetaires posées à moins de 4 cases de l'ennemi -> mortes avant de tirer.
            x, y, gap = self._ranged_spot(me["range"], group)
            if gap is None:
                return None                              # aucune case sûre : on ne la jette pas dans la mêlée
            where = f"à {gap:.1f} cases de l'ennemi le plus proche (portée {me['range']:.1f})"
            verdict = f"gagne en ~{t_kill:.0f} s" if not lose else "ralentit seulement"
            return Decision(card, playable[card], x, y,
                            f"défense : {t.name} x{len(group)} -> {card} [stats : {verdict}, {where}]", precise=True,
                            trade=self._trade(card, group, near_tower, push_cost), push=push_cost)
        if lose:
            # combat difficile : devant notre tour, pour que tour et unité frappent ensemble
            lane_x = LANES_X[_lane(t.x)]
            x, y = lane_x + (0.06 if lane_x < 0.5 else -0.06), OWN_TOWER_Y - 0.05
            where = "devant notre tour (tour + unité)"
        else:
            x, y = t.x + t.vx * 0.5, t.y + 0.05
            where = "au contact"
        x, y = _clamp_own(x, y)
        verdict = f"gagne en ~{t_kill:.0f} s" if not lose else "ralentit seulement"
        return Decision(card, playable[card], x, y,
                        f"défense : {t.name} x{len(group)} -> {card} [stats : {verdict}, {where}]", precise=True,
                        trade=self._trade(card, group, near_tower, push_cost), push=push_cost)

    def _defend(self, threats: list[Seen], playable: dict) -> Decision | None:
        # la menace la plus proche de nos tours (le plus bas à l'écran)
        t = max(threats, key=lambda s: s.y)
        lane_x = LANES_X[_lane(t.x)]
        is_air, is_tank = t.name in AIR_UNITS, t.name in TANK_UNITS
        swarm = sum(1 for s in threats if s.name in SWARM_UNITS and math.hypot(s.x - t.x, s.y - t.y) < 0.15)
        if is_air and t.name not in ("balloon", "lava-hound"):   # ceux-là ignorent les troupes : pas d'appât
            anti_air = [c for c in playable if c in DECK and DECK[c].kind == "troop" and card_info.combat(c)["hits_air"]]
            decoy = next((c for c in ("valkyrie", "knight") if c in playable), None)
            if not anti_air and decoy and t.y < OWN_TOWER_Y - 0.03:
                # rien ne tire en l'air : un tank au sol sert d'APPÂT. Pas collé à l'unité volante, mais décalé
                # vers le centre (~3 cases) et un peu devant la tour : elle doit faire un détour pour l'atteindre,
                # pendant ce temps nos deux tours la frappent, et l'appât encaisse moins longtemps
                side = 1 if lane_x < 0.5 else -1
                x, y = _clamp_own(lane_x + side * 3 * PHONE.tw, max(t.y + 0.04, OWN_TOWER_Y - 3 * PHONE.th))
                push_cost = sum(_cost(n) for n in {s.name for s in threats if math.hypot(s.x - t.x, s.y - t.y) < 0.2})
                if push_cost < 3:
                    return None             # Chauves-souris seules (2) : la tour s'en charge, pas d'appât à 3-4
                return Decision(decoy, playable[decoy], x, y,
                                f"défense : {t.name} (volant, rien pour tirer en l'air) -> {decoy} en appât décalé",
                                precise=True, push=push_cost)
        if is_air:
            order = ["witch", "musketeer", "archers", "minions", "spear-goblins"]   # la Sorcière : air + zone
        elif is_tank:
            order = ["mini-pekka", "witch", "musketeer", "knight", "valkyrie", "minions", "archers"]   # squelettes = appâts
        elif swarm >= 2:
            order = ["valkyrie", "witch", "knight", "musketeer", "archers", "mini-pekka"]   # dégâts de zone
        else:
            order = ["knight", "valkyrie", "mini-pekka", "musketeer", "archers", "minions"]
        # échange d'élixir : parmi les cartes adaptées, ne pas payer plus que l'attaque (+1) si une moins chère suffit
        push_cost = sum(_cost(n) for n in {s.name for s in threats if math.hypot(s.x - t.x, s.y - t.y) < 0.2})
        group = [s for s in threats if _tile_dist(s.x, s.y, t.x, t.y) < 4]
        near_tower = t.y > OWN_TOWER_Y - 0.1
        # une troupe VOLANTE que la menace ne peut pas toucher, et qui gagne : réponse parfaite, avant le Canon
        # (29/09 : P.E.K.K.A défendu 5 fois au Canon + Chevalier alors que nos Gargouilles le tuent sans perte)
        if not is_air and not any(card_info.combat(g.name)["hits_air"] for g in group):
            immune = {c: i for c, i in playable.items() if c in DECK and DECK[c].kind == "troop"
                      and card_info.combat(c)["flying"] and (self._duel(c, group, near_tower) or (False,))[0]}
            if immune:
                d = self._stat_pick(t, threats, immune, push_cost)
                if d:
                    return d
        if self.evo_musketeer() and "musketeer" in playable and len(group) >= 2 and not is_air and t.name != "balloon":
            # Mousquetaire ÉVOLUÉE avant le Canon : ses 3 tirs de sniper percent toute la file (tank + soutien)
            lane_x = LANES_X[_lane(t.x)]
            x, y = _clamp_own(lane_x, OWN_TOWER_Y + 0.06)
            return Decision("musketeer", playable["musketeer"], x, y,
                            f"défense : {t.name} x{len(group)} -> Mousquetaire évoluée derrière la tour (sniper le long du couloir)",
                            precise=True, trade=self._trade("musketeer", group, True, push_cost), push=push_cost)
        # Canon : le bâtiment au centre attire les tanks (Géant, Hog…) entre les deux tours
        if "cannon" in playable and not is_air and (is_tank or t.name in FAST_BUILDING_HUNTERS or t.name in SINGLE_MELEE
                                                     or _cost(t.name) >= 3)                 and len(group) < 3 and card_info.combat(t.name)["count"] < 3:
            # pas contre les nuées (5 Barbares) : le Canon tire sur 1 cible à la fois. D'après la CARTE, pas d'après ce
            # que voit le scanner (29/09 : 1 Barbare vu sur 5 -> Canon posé 4 fois)
            col = 8 if lane_x < 0.5 else 9             # centre, côté de la menace
            if t.name in FAST_BUILDING_HUNTERS:
                row, why = OWN_FIRST_ROW + 4, "4 cases sous la rivière : le Cochon est dévié entre les 2 tours"
                if set(self.opp_hand) & BIG_SPELLS:
                    # il a Boule de feu/Poison en main : il visera l'emplacement habituel -> on le pose plus haut
                    row, why = OWN_FIRST_ROW + 2, "plus haut que d'habitude : son sort prédit tombera à côté"
            elif t.name == "royal-giant":
                row, why = OWN_FIRST_ROW + 2, "près de la rivière : le Géant royal tire de loin, il faut l'attirer tôt"
            elif t.name in SLOW_TANKS:
                # pas plus bas : trop en retrait, le tank ne « voit » pas le Canon et file sur la tour (29/09)
                row, why = OWN_FIRST_ROW + 3, "3 cases sous la rivière : assez près pour attirer le tank au centre"
            else:
                row, why = OWN_FIRST_ROW + 4, "4 cases sous la rivière"
            x, y = PHONE.center(col, row)
            return Decision("cannon", playable["cannon"], x, y, f"défense : {t.name} -> canon ({why})", precise=True,
                            trade=self._trade("cannon", group, True, push_cost), push=push_cost)
        if self.p.get("stat_defense", True):
            d = self._stat_pick(t, threats, playable, push_cost)
            if d:
                return d
        fits = [c for c in order if c in playable and c in DECK]
        # une nuée fragile (Gargouilles) contre des dégâts de zone qui touchent sa couche (Sorcier, Dragons squelettes,
        # Bourreau…) meurt d'un coup : jamais, même en secours (29/09 : Gargouilles sur un Sorcier)
        def splashed(c):
            me = card_info.combat(c)
            return me["count"] >= 3 and any(
                card_info.combat(g.name)["splash"] and (card_info.combat(g.name)["hits_air"] if me["flying"]
                                                        else card_info.combat(g.name)["hits_ground"]) for g in group)
        fits = [c for c in fits if not splashed(c)]
        cheap = [c for c in fits if DECK[c].cost <= push_cost + 1]
        for card in cheap or sorted(fits, key=lambda c: DECK[c].cost):   # sinon, au moins la moins chère
            c = DECK[card]
            if t.name in SINGLE_MELEE and c.kind == "troop" and c.targets == "ground" and t.y < OWN_TOWER_Y - 0.04:
                # mêlée mono-cible : au centre entre les tours, l'ennemi dévie vers le milieu et les deux tours tirent
                x, y = _clamp_own(0.5 + (-0.04 if lane_x < 0.5 else 0.04), OWN_TOWER_Y - 0.03)
                return Decision(card, playable[card], x, y, f"défense : {t.name} -> {card} au centre (les 2 tours tirent)",
                                precise=True, trade=self._trade(card, group, True, push_cost), push=push_cost)
            ranged = c.targets == "air+ground" and not c.flying and card_info.combat(card)["range"] >= 4
            if ranged:
                # tireur : même règle que la défense par les stats (case sûre, jamais au contact), et le modèle des
                # pros ne la déplace pas (30/09 : Mousquetaire posée à 1,3 case d'un ennemi par le modèle)
                x, y, gap = self._ranged_spot(card_info.combat(card)["range"], group)
                if gap is None:
                    x, y = lane_x + (0.14 if lane_x < 0.5 else -0.14), OWN_TOWER_Y + 0.085   # derrière la tour
            elif c.targets == "air+ground" and not c.flying:
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
            return Decision(card, playable[card], x, y, f"défense : {t.name} ({kind}) -> {card}", precise=ranged,
                            trade=self._trade(card, group, near_tower, push_cost), push=push_cost)
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
        mode = self._endgame(now)
        # il vient de poser une carte lourde : il est à sec, on frappe tout de suite dans l'AUTRE couloir
        if (self.p.get("punish_opposite") and mode != "défendre l'avance" and now - self.opp_heavy_t < 4
                and elixir >= 7):          # 4 pour la carte + 3 gardés pour défendre l'autre côté
            for card in ("mini-pekka", "knight"):
                if card in playable:
                    lane = self._weak_lane(seen)
                    x, y = _clamp_own(LANES_X[lane], 0.47)
                    return Decision(card, playable[card], x, y, f"punition : carte lourde en face -> {card} au pont opposé",
                                    precise=True)
        # la barre d'élixir se lit au plus ~9.5 : un seuil à 10 ne serait jamais atteint ; et à élixir plein le
        # Géant passe avant la carte qu'on ferait « tourner » (match du 27/09 : Géant jamais joué)
        giant_at = min(self.p["giant_elixir"], 9) - (1 if fast else 0)
        # ses contres au Géant connus et tous hors de sa main (joués récemment) : fenêtre pour lancer plus tôt
        known = set(self.opp_deck) & GIANT_COUNTERS
        if self.p.get("giant_when_counter_out") and known and not known & set(self.opp_hand):
            giant_at -= 2
        edge, gap = self._edge(elixir)
        if edge > 0:
            giant_at = min(giant_at, 6)      # il ne peut pas tout défendre : on attaque dès qu'on peut suivre
        elif edge < 0:
            giant_at = max(giant_at, 9)      # en retard : on défend à l'économie, pas d'attaque
        if mode == "défendre l'avance":
            playable = {c: i for c, i in playable.items() if c != "giant"}   # plus d'attaque : on garde nos couronnes
        elif mode == "tout pour l'attaque":
            giant_at = min(giant_at, 5)      # mené, peu de temps : chaque seconde sans pression est perdue
        elif mode == "mort subite":
            giant_at = min(giant_at, 7)      # la première tour qui tombe gagne
        if elixir >= self._cycle_at(now):
            giant_at = min(giant_at, elixir)  # jamais d'élixir perdu, même en retard
        if mode not in ("tout pour l'attaque", "mort subite"):
            # un Géant seul se fait démonter : on le pose seulement si, après ses 5 élixirs, il en reste pour le
            # SOUTIEN derrière lui (3) — la réserve de défense se reconstitue pendant qu'il marche jusqu'au pont
            giant_at = max(giant_at, GIANT_FOLLOW_ELIXIR - (1 if fast else 0))
            punish = punish and elixir >= GIANT_FOLLOW_ELIXIR - 1
        if "giant" in playable and (elixir >= giant_at or punish):
            lane = self._attack_lane(seen, now if self.p["counter_push"] else None)
            tower_down = self.match is not None and not all(self.match.enemy_alive.values())
            spot = self.p["giant_spot"]
            if spot == "back" and set(self.opp_deck) & PULL_BUILDINGS:
                spot = "corner"     # son bâtiment ne pourra pas tirer le Géant vers le centre depuis le coin
            lx = LANES_X[lane] + ((-0.12 if lane == 0 else 0.12) if spot == "corner" else 0)
            if spot == "king":
                lx = 0.5 + (-0.12 if lane == 0 else 0.12)   # juste derrière le Roi, du côté du couloir visé
            x, y = _clamp_own(lx, {"king": 0.69, "back": 0.69, "corner": 0.69, "mid": 0.55, "bridge": 0.47}.get(spot, 0.69))
            countered_out = self.p.get("giant_when_counter_out") and known and not known & set(self.opp_hand)
            why = mode if mode and elixir < self.p["giant_elixir"] else \
                "l'ennemi est à sec" if punish and elixir < giant_at else \
                f"avance d'élixir {gap:+.1f}" if edge > 0 and elixir < self.p["giant_elixir"] else \
                "ses contres sont joués" if countered_out and elixir < self.p["giant_elixir"] else \
                "double élixir" if fast and elixir < self.p["giant_elixir"] else \
                "sa tour de ce côté est tombée" if tower_down and lane == self._attack_lane(seen) else \
                {"back": "au fond", "corner": "dans le coin" + (" (il a un bâtiment)" if set(self.opp_deck) & PULL_BUILDINGS else ""),
                 "mid": "au milieu", "bridge": "au pont", "king": "derrière le Roi"}.get(spot, spot)
            if punish and elixir < giant_at:
                x, y = _clamp_own(LANES_X[lane], 0.47)   # pression immédiate au pont
                return Decision("giant", playable["giant"], x, y, f"attaque : Géant ({why}) au pont", precise=True)
            return Decision("giant", playable["giant"], x, y, f"attaque : Géant ({why})")
        # soutien derrière notre Géant pendant qu'il avance
        ours = [s for s in seen if not s.enemy and s.name == "giant"]
        support_at = self.p["support_min_elixir"] + (-1 if edge > 0 else 2 if edge < 0 else 0)
        if mode in ("tout pour l'attaque", "mort subite"):
            support_at = min(support_at, 3)
        if ours and elixir >= support_at and mode != "défendre l'avance":
            g = ours[0]
            order = ["witch", "musketeer", "archers", "valkyrie", "mini-pekka", "minions"]   # Sorcière derrière le Géant
            if set(self.opp_hand) & (SMALL_SPELLS | BIG_SPELLS):   # Boule de feu / Poison tuent aussi les archères
                # ses Flèches/Zap/Bûche tueraient archères ou gargouilles : on les passe en dernier
                order = [c for c in order if c not in ("archers", "minions")] + ["archers", "minions"]
            # garder de quoi défendre l'AUTRE couloir (Chevalier/Canon = 3) : s'il contre-attaque de l'autre côté
            # pendant notre poussée, on ne doit pas être à sec. Sauf s'il est lui-même à sec ou en fin de match
            reserve = 0 if self.opp_elixir < 3 or mode in ("tout pour l'attaque", "mort subite") else DEFENSE_RESERVE
            for card in order:
                # la Sorcière : ses squelettes défendent aussi -> 1 élixir de réserve en moins (30/09 : jamais jouée,
                # 5 + 3 = 8 élixirs requis alors que le soutien se décide vers 7)
                if card in playable and elixir - DECK[card].cost >= reserve - (1 if card == "witch" else 0):
                    x, y = _clamp_own(g.x, g.y + 0.07)       # ~4 cases derrière : hors d'une Boule de feu sur le Géant
                    return Decision(card, playable[card], x, y, f"soutien : {card} derrière le Géant")
        d = self._counter_push(seen, playable, elixir, now, support_at, mode, edge)
        if d:
            return d
        if elixir >= self._cycle_at(now) and playable:
            # élixir plein et rien à faire : on fait tourner la carte la moins chère, sans risque
            card = min((c for c in playable if DECK[c].kind == "troop"), key=lambda c: DECK[c].cost, default=None)
            if "witch" in playable:
                card = "witch"          # élixir plein : c'est LE moment de la Sorcière (5), sinon elle bloque la main
            if card == "archers":
                # au centre juste devant le Roi : les deux archères partent chacune dans un couloir,
                # une seule Flèche ne peut plus les prendre ensemble
                x, y = _clamp_own(0.5, OWN_TOWER_Y + 0.0)
                return Decision(card, playable[card], x, y, "rien à faire : archères séparées au centre", precise=True)
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
