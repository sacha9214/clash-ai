"""Vitesse des unités suivies, mesurée sur l'horloge réelle.

Sans dépendance lourde : importé par detect_yolo.py (ultralytics 8.4) ET detect_katacr.py (ultralytics 8.1).

Les images n'arrivent pas à intervalle régulier : la boucle de décision, plus des rafales pendant la pose d'une carte
(agent._observe). La vitesse vient donc de positions HORODATÉES : droite des moindres carrés sur les WINDOW_S
dernières secondes (pente = vitesse). Plus robuste qu'une moyenne de différences : la gigue des boîtes (1-3 px) n'est
jamais divisée par un petit écart de temps, et une rafale ne pèse pas plus qu'une image normale (MERGE_S).
"""
from __future__ import annotations


def feet(box) -> tuple[float, float]:
    """Point suivi : milieu du bas de la boîte, les pieds de l'unité (la position de l'unité dans brain.py)."""
    return (box[0] + box[2]) / 2, float(box[3])


class Motion:
    """Vitesse d'un point suivi, en px/s (x vers la droite, y vers le bas).

    - observations à moins de MERGE_S d'écart : fusionnées (moyenne), une rafale compte pour un point ;
    - (0, 0) tant que les points ne couvrent pas MIN_SPAN_S : deux boîtes à 30 ms d'écart ne donnent que du bruit ;
    - on garde le dernier point d'avant la fenêtre : après un trou (unité masquée) la vitesse reste mesurable
      (vitesse moyenne sur le trou) ; trou de plus de MAX_GAP_S : on repart de zéro, l'ancienne vitesse est périmée.
    """
    WINDOW_S = 0.7
    MERGE_S = 0.03
    MIN_SPAN_S = 0.15
    MAX_GAP_S = 1.0

    def __init__(self):
        self.pts: list[list[float]] = []     # paquets [instant de la 1re observation, instant moyen, x, y, nombre]
        self.vel: tuple[float, float] = (0.0, 0.0)
        self.last_t = float("-inf")          # instant de la dernière observation

    def update(self, t: float, x: float, y: float) -> tuple[float, float]:
        p = self.pts
        if p and t - self.last_t > self.MAX_GAP_S:
            p.clear()
        self.last_t = t
        if p and t - p[-1][0] < self.MERGE_S:
            q = p[-1]
            q[4] += 1
            q[1] += (t - q[1]) / q[4]
            q[2] += (x - q[2]) / q[4]
            q[3] += (y - q[3]) / q[4]
        else:
            p.append([t, t, x, y, 1])
        while len(p) > 2 and p[1][1] <= t - self.WINDOW_S:
            p.pop(0)
        self.vel = self._fit()
        return self.vel

    def _fit(self) -> tuple[float, float]:
        p = self.pts
        if len(p) < 2 or p[-1][1] - p[0][1] < self.MIN_SPAN_S:
            return 0.0, 0.0
        t0 = p[-1][1]                        # temps relatifs : pas de perte de précision sur perf_counter()
        n = len(p)
        tm = sum(q[1] - t0 for q in p) / n
        xm = sum(q[2] for q in p) / n
        ym = sum(q[3] for q in p) / n
        stt = sum((q[1] - t0 - tm) ** 2 for q in p)
        return (sum((q[1] - t0 - tm) * (q[2] - xm) for q in p) / stt,
                sum((q[1] - t0 - tm) * (q[3] - ym) for q in p) / stt)


class Tracks(dict):
    """Suivi -> Motion, pour toutes les unités suivies ; oublié KEEP_S secondes après la dernière image où il a été vu."""
    KEEP_S = 2.0

    def observe(self, units, t: float) -> None:
        """Mesure la vitesse de chaque unité suivie (track_id >= 0) et la range dans u.vel."""
        for u in units:
            if u.track_id >= 0:
                u.vel = self.setdefault(u.track_id, Motion()).update(t, *feet(u.box))

    def forget(self, t: float) -> None:
        for tid in [tid for tid, m in self.items() if t - m.last_t > self.KEEP_S]:
            del self[tid]
