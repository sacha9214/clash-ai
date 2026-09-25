"""Boucle de combat du cerveau v0 : voir -> comprendre -> décider -> jouer (vérifié)."""
from __future__ import annotations

import json
import os
import queue
import threading
import time
import unicodedata

import cv2

from clashai import battle as B
from clashai import hand as H
from clashai.actions import play_card
from clashai.brain import Brain
from clashai.detect import Detector, draw
from clashai.opponent import UNIT2CARD, Opponent
from clashai.towers import MatchState
import numpy as np


def _ascii(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


# Écriture des captures en arrière-plan : cv2.imwrite ne doit pas bloquer la boucle de combat
_jobs: queue.Queue = queue.Queue()


def _writer():
    while True:
        path, img = _jobs.get()
        cv2.imwrite(path, img)
        _jobs.task_done()


threading.Thread(target=_writer, daemon=True).start()


def _save(path: str, img) -> None:
    _jobs.put((path, img))


class Agent:
    def __init__(self, out: str = "runs/games", show: bool = False):
        self.show = show
        self.det, self.brain, self.out = Detector(track=True), Brain(), out
        self.opp, self.opp_log = Opponent(), []
        self.spells_pending, self.spells_log = [], []
        self.perf, self.perf_log = {"det_ms": 0.0}, []
        self.match = MatchState()
        os.makedirs(out, exist_ok=True)

    def think(self, img, now, fps):
        t0 = time.perf_counter()
        units = self.det(img)
        self.perf["det_ms"] = (time.perf_counter() - t0) * 1000
        self._watch_spells(units, img, now)
        self.match.update(img, units, now)
        self.brain.match = self.match
        h, w = img.shape[:2]
        seen = Brain.to_seen(units, w, h, self.det.trails, fps)
        hand = H.read_hand(img, min_score=0.45)
        ready = [B.card_ready(img, s) for s in range(4)]
        self._learn_new_card(img, hand, ready)
        hand = H.read_hand(img, min_score=0.45)
        el = B.read_elixir(img)
        new = self.opp.update(units, now, img.shape[0])
        for c in new:
            self.opp_log.append({"t": round(now, 2), "card": c, "elixir_after": round(self.opp.elixir, 1)})
        self.brain.opp_elixir = self.opp.elixir
        self.brain.opp_hand, self.brain.opp_deck = self.opp.hand, self.opp.deck
        costs = {card: cost for card, cost, _ in UNIT2CARD.values()}
        if any(costs.get(c, 0) >= 6 for c in new):
            self.brain.opp_heavy_t = now
        d = self.brain.decide(seen, hand, ready, el, now)
        info = [f"elixir {el:.1f}  main : " + ", ".join(c or "?" for c in hand), self.match.summary()]
        return units, d, info, hand, el

    def _show(self, img, units, d, info):
        if self.show:
            view = self.annotate(img, units, d, info)
            cv2.imshow("Clash AI", cv2.resize(view, (int(view.shape[1] * 1.25), int(view.shape[0] * 1.25))))
            cv2.waitKey(1)

    def _watch_spells(self, units, img, now):
        """Mesure le temps de vol de nos sorts : du tap jusqu'à ce que le détecteur voie l'effet près de la cible.
        Sert à caler SPELL_IMPACT_S (brain.py) sur de vrais matchs."""
        from clashai.brain import KING_Y, _tile_dist
        h, w = img.shape[:2]
        for sp in list(self.spells_pending):
            if now - sp["t_tap"] > 4:
                self.spells_pending.remove(sp)            # jamais vu : on abandonne
                continue
            for u in units:
                if u.name == sp["card"] and _tile_dist(u.center[0] / w, u.center[1] / h, sp["x"], sp["y"]) < 4:
                    self.spells_log.append({"card": sp["card"], "flight_s": round(now - sp["t_tap"], 2),
                                            "dist_tiles": round(_tile_dist(sp["x"], sp["y"], 0.5, KING_Y), 1)})
                    self.spells_pending.remove(sp)
                    break

    def _observe(self, img, info):
        """Pendant qu'une carte se pose : on continue de suivre les unités et d'afficher (pas de décision)."""
        now = time.time()
        units = self.det(img)
        self._watch_spells(units, img, now)
        for c in self.opp.update(units, now, img.shape[0]):
            self.opp_log.append({"t": round(now, 2), "card": c, "elixir_after": round(self.opp.elixir, 1)})
        self._show(img, units, None, info)

    def _learn_new_card(self, img, hand, ready):
        """Une carte du deck sans exemple (nouvelle dans le deck) : quand un emplacement
        en couleur reste illisible 3 images de suite, c'est elle -> on l'apprend."""
        from clashai.cards import DECK
        missing = [c for c in DECK if c not in H.known_cards()]
        if len(missing) != 1:
            return
        streak = getattr(self, "_unknown_streak", {})
        for slot in range(4):
            crop = H.card_crop(img, slot)
            name, score = H.identify(crop)
            if hand[slot] is None and ready[slot] and score < 0.45:
                streak[slot] = streak.get(slot, 0) + 1
                if streak[slot] >= 3:
                    H.learn(crop, missing[0])
                    print(f"   carte apprise : {missing[0]} (score {score:.2f})", flush=True)
                    streak.clear()
                    break
            else:
                streak[slot] = 0
        self._unknown_streak = streak

    def annotate(self, img, units, d, info):
        v = img.copy()
        # bandeau du haut : ce que l'IA sait de l'adversaire
        h, w = v.shape[:2]
        v[0:78] = (v[0:78] * 0.2).astype(v.dtype)
        for i, line in enumerate(self.opp.banner()):
            cv2.putText(v, _ascii(line), (6, 20 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.47,
                        (80, 200, 255) if i == 0 else (255, 255, 255), 1, cv2.LINE_AA)
        draw(v, units, self.det.trails)
        if d:
            h, w = v.shape[:2]
            p = (int(d.x * w), int(d.y * h))
            cv2.circle(v, p, 16, (0, 255, 255), 3)
            cv2.arrowedLine(v, B.px(v, B.CARD_X[d.slot], B.CARD_Y - 0.05), p, (0, 255, 255), 2)
            info = [d.reason] + info
        for i, line in enumerate(info):
            line = _ascii(line)
            cv2.putText(v, line, (8, 100 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(v, line, (8, 100 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return v

    def play_battle(self, dev, game_id: str, params: dict | None = None) -> dict:
        """Joue un combat jusqu'au bout avec la variante de stratégie `params`."""
        self.brain = Brain(params)
        self.opp, self.opp_log = Opponent(), []
        self.spells_pending, self.spells_log = [], []
        self.perf, self.perf_log = {"det_ms": 0.0}, []
        self.match = MatchState()
        folder = os.path.join(self.out, game_id)
        os.makedirs(folder, exist_ok=True)
        self.det.tracker.reset() if self.det.tracker is not None else None
        log, last_play, gone, t_prev, n, refused = [], 0.0, None, time.time(), 0, 0
        while True:
            img, t_recv, _ = dev.frame()
            t_loop = time.perf_counter()
            if not B.in_battle(img):
                gone = gone or time.time()
                if time.time() - gone > 5:
                    break
                time.sleep(0.2)
                continue
            gone = None
            now = time.time()
            fps, t_prev = 1 / max(now - t_prev, 1e-3), now
            units, d, info, hand, el = self.think(img, now, fps)
            t_show = time.perf_counter()
            self._show(img, units, d, info)
            # latence de chaque tour de boucle : âge de l'image, détection, reste de la réflexion, affichage
            end = time.perf_counter()
            self.perf_log.append({"age_ms": round((t_loop - t_recv) * 1000), "det_ms": round(self.perf["det_ms"]),
                                  "think_ms": round((t_show - t_loop) * 1000 - self.perf["det_ms"]),
                                  "show_ms": round((end - t_show) * 1000), "loop_ms": round((end - t_loop) * 1000)})
            if now - getattr(self, "_last_raw", 0) > 1.5:
                # image brute (sans dessins) : matière pour étiqueter nos propres images (phase C du détecteur)
                self._last_raw = now
                raw_dir = os.path.join(self.out, "..", "capture", game_id)
                os.makedirs(raw_dir, exist_ok=True)
                _save(os.path.join(raw_dir, f"{int(now * 10) % 10**7:07d}.jpg"), img.copy())
            if now - getattr(self, "_last_snap", 0) > 5:
                self._last_snap = now
                _save(os.path.join(folder, f"state{int(now) % 100000:05d}.jpg"), self.annotate(img, units, None, info))
            if d and now - last_play > 0.8:
                h, w = img.shape[:2]
                t_play = time.perf_counter()
                ok = play_card(dev, d.slot, (int(d.x * w), int(d.y * h)),
                               on_frame=lambda im: self._observe(im, info))
                play_ms = round((time.perf_counter() - t_play) * 1000)
                if ok and d.card in ("arrows", "fireball"):
                    self.opp.note_our_spell(time.time(), d.x * w, d.y * h)
                    self.spells_pending.append({"card": d.card, "x": d.x, "y": d.y, "t_tap": time.time()})
                log.append({"t": round(now, 2), "card": d.card, "x": round(d.x, 3), "y": round(d.y, 3), "tile": d.tile,
                            "reason": d.reason, "ok": ok, "play_ms": play_ms, "elixir": el, "hand": hand,
                            "units": [(u.name, u.enemy, u.center) for u in units]})
                if ok:
                    last_play = time.time()
                    _save(os.path.join(folder, f"play{n:03d}.jpg"), self.annotate(img, units, d, info))
                    n += 1
                else:
                    refused += 1
                    last_play = time.time() - 0.4
        _jobs.join()   # captures en attente écrites avant le résumé
        with open(os.path.join(folder, "perf.jsonl"), "w") as f:
            for row in self.perf_log:
                f.write(json.dumps(row) + chr(10))
        with open(os.path.join(folder, "spells.jsonl"), "w") as f:   # temps de vol mesurés de nos sorts
            for row in self.spells_log:
                f.write(json.dumps(row) + "\n")
        with open(os.path.join(folder, "decisions.jsonl"), "w") as f:
            for row in log:
                f.write(json.dumps(row, default=str) + "\n")
        # résumé de fin de match : l'adversaire tel que l'IA l'a compris
        card = np.zeros((170, 578, 3), np.uint8)
        lines = [f"Deck adverse ({len(self.opp.deck)}/8 vues) :", ", ".join(self.opp.deck[:4]),
                 ", ".join(self.opp.deck[4:8]), f"Cartes jouees : {len(self.opp.played)}   elixir depense ~" +
                 str(sum(next(v[1] for v in UNIT2CARD.values() if v[0] == c) for c in self.opp.played))]
        for i, line in enumerate(lines):
            cv2.putText(card, _ascii(line), (10, 32 + 38 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(folder, "opponent_summary.jpg"), card)
        with open(os.path.join(folder, "opponent.json"), "w") as f:
            json.dump({"played": self.opp_log, "deck": self.opp.deck}, f, indent=1)
        return {"game": game_id, "played": n, "refused": refused, "params": self.brain.p,
                "enemy_deck": self.opp.deck}
