"""Grille de l'arène : 18 cases de large x 32 de haut, comme dans le jeu.

Le jeu pose tout sur ces cases. On parle en cases (tx, ty) : tx 0 à gauche, ty 0 en haut
(fond ennemi) ; la rivière occupe les rangées 15-16, nos troupes se posent en 17-31.
La caméra est inclinée : une case paraît ~0,7x moins haute que large, d'où deux échelles.

Deux calibrations :
- PHONE : fractions de l'écran du REDMAGIC (mesures de brain.py : rivière 0.425, tours
  0.59 / 0.66, couloirs 0.205 / 0.795 ; le fond de la grille tombe sur la clôture à 0.705).
- ARENA : fractions de l'arène recadrée 568x896 des vidéos / de KataCR (médianes des
  étiquettes de tours du jeu de données : princesses x 0.20 / 0.805, rivière au milieu).
"""
from __future__ import annotations

from dataclasses import dataclass

COLS, ROWS = 18, 32
RIVER_ROWS = (15, 16)
OWN_FIRST_ROW = 17


@dataclass(frozen=True)
class Grid:
    x0: float   # bord gauche de la case 0
    tw: float   # largeur d'une case
    y0: float   # bord haut de la rangée 0
    th: float   # hauteur d'une case

    def to_tile(self, x: float, y: float) -> tuple[float, float]:
        """Fractions d'image -> cases (flottants ; int() pour la case qui contient le point)."""
        return (x - self.x0) / self.tw, (y - self.y0) / self.th

    def to_frac(self, tx: float, ty: float) -> tuple[float, float]:
        return self.x0 + tx * self.tw, self.y0 + ty * self.th

    def center(self, col: int, row: int) -> tuple[float, float]:
        return self.to_frac(col + 0.5, row + 0.5)

    def cell(self, x: float, y: float) -> tuple[int, int]:
        tx, ty = self.to_tile(x, y)
        return min(max(int(tx), 0), COLS - 1), min(max(int(ty), 0), ROWS - 1)

    def snap(self, x: float, y: float) -> tuple[float, float]:
        """Centre de la case qui contient le point : là où le jeu posera vraiment la carte."""
        return self.center(*self.cell(x, y))


PHONE = Grid(x0=0.205 - 3.5 * (0.59 / 11), tw=0.59 / 11, y0=0.425 - 16 * 0.01737, th=0.01737)
ARENA = Grid(x0=0.2 - 3.5 * 0.055, tw=0.055, y0=0.5 - 16 * 0.0278, th=0.0278)
