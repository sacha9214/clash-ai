"""Identité d'une unité suivie : son camp et son nom, votés sur TOUTE sa vie (pas image par image).

Sans dépendance lourde (importé par les détecteurs, testé hors GPU).

- Camp : on ne pose une troupe que dans SA moitié -> une unité née loin de la rivière a un avis de départ
  (née en haut = ennemie). Ensuite chaque image vote : couleur du badge / barre de vie (forte, 3) et avis du
  détecteur (faible, sa confiance), sur une fenêtre glissante : une erreur des premières images se corrige
  (avant : vote figé après 8 images, une erreur durait toute la vie de l'unité).
- Nom : somme des confiances par nom sur toute la vie du suivi (un Chevalier lu « Valkyrie » sur une image reste
  Chevalier). Nos unités ne peuvent être que celles de notre deck : le meilleur nom permis l'emporte.
"""
from __future__ import annotations

import collections

BIRTH_PRIOR = 2.0        # poids de l'avis « né chez lui / chez nous »
BIRTH_MARGIN = 0.05      # née à moins de 5 % de la rivière : pas d'avis (ça traverse)
COLOR_WEIGHT = 3.0
WINDOW = 30              # votes gardés (~1-2 s de jeu)


class TrackIdentity:
    def __init__(self, birth_y: float, river_y: float = 0.5):
        """birth_y, river_y : fractions de hauteur de l'arène (0 = fond ennemi)."""
        d = birth_y - river_y
        self.prior = 0.0 if abs(d) < BIRTH_MARGIN else (BIRTH_PRIOR if d < 0 else -BIRTH_PRIOR)
        self.votes: collections.deque = collections.deque(maxlen=WINDOW)
        self.names: dict[str, float] = collections.defaultdict(float)
        self.last_t = 0.0

    def observe(self, name: str, conf: float, detector_enemy: bool, color: int, t: float = 0.0) -> None:
        """color : +1 badge rouge (ennemi), -1 bleu (nous), 0 illisible."""
        self.votes.append(COLOR_WEIGHT * color if color else (conf if detector_enemy else -conf))
        self.names[name] += conf
        self.last_t = t

    @property
    def enemy(self) -> bool:
        return self.prior + sum(self.votes) > 0

    def name(self, allowed: set[str] | None = None) -> str:
        """Nom le plus voté ; parmi `allowed` (notre deck) si l'un d'eux a des votes."""
        pool = {n: s for n, s in self.names.items() if allowed is None or n in allowed} or self.names
        return max(pool, key=pool.get)
