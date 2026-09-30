"""Boucle de combat du cerveau v0 : voir -> comprendre -> décider -> jouer (vérifié)."""
from __future__ import annotations

import json
import os
import queue
import threading
import time
import unicodedata

import cv2
from pathlib import Path

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


COSTS = {card: cost for card, cost, _ in UNIT2CARD.values()}


class StreamLost(RuntimeError):
    """Le flux vidéo ne donne plus d'image : téléphone débranché ou serveur scrcpy arrêté."""


LIVE_DIR = Path(__file__).resolve().parents[1] / "runs"
EMOTE_CHAT = (0.097, 0.89)
LIVE_EVERY_S = 0.12            # ~8 images/s pour la page de suivi (fil d'affichage, pas celui des décisions)     # bulle de discussion, en bas à gauche pendant un combat   # runs/live.jpg : suivi à distance

class Agent:
    ctx: str | None = None                       # type du deck adverse de la partie en cours
    STALL_S = 5.0          # plus aucune nouvelle image depuis ce délai : le flux est mort

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
        raw = self.det(img)
        self.perf["det_ms"] = (time.perf_counter() - t0) * 1000
        # l'adversaire apprend son deck sur toutes les détections ; le cerveau ne voit que les unités crédibles
        units = [u for u in raw if self.opp.plausible(u)]
        self._watch_spells(units, img, now)
        self.match.update(img, units, now)
        self.brain.match = self.match
        h, w = img.shape[:2]
        seen = Brain.to_seen(units, w, h, self.det.trails, fps)
        hand = H.read_hand(img, min_score=0.45)
        ready = [B.card_ready(img, s) for s in range(4)]
        if self._learn_new_card(img, hand, ready):
            hand = H.read_hand(img, min_score=0.45)
        el = B.read_elixir(img)
        self._opp_update(raw, now, img.shape[0])
        self.brain.opp_elixir = self.opp.elixir
        self.brain.opp_hand, self.brain.opp_deck = self.opp.hand, self.opp.deck
        if self.ctx is None and len(self.opp.deck) >= 4:
            # son type de deck est reconnu : on reprend la stratégie qui a le mieux marché contre ce type
            from clashai import strategy as ST
            self.ctx = ST.archetype(self.opp.deck)
            self.brain.p.update(ST.choose_for(self.ctx, ST.load(), ST.load_ctx()))
            print(f"   adversaire de type « {self.ctx} » : stratégie apprise contre ce type", flush=True)
        d = self.brain.decide(seen, hand, ready, el, now)
        info = [f"elixir {el:.1f}  main : " + ", ".join(c or "?" for c in hand), self.match.summary(),
                f"avance {el - self.opp.elixir:+.1f} (lui ~{self.opp.elixir:.1f})  echanges (estime) {self.brain.trade_balance:+.1f}"]
        return units, d, info, hand, el

    def _show(self, img, units, d, info):
        """La fenêtre tourne dans son propre fil : si Windows la bloque (déplacement, clic), l'IA continue de jouer."""
        if not self.show:
            return
        self._view = (img, units, d, info)
        if not getattr(self, "_viewer", None):
            self._viewer = threading.Thread(target=self._viewer_loop, daemon=True)
            self._viewer.start()

    def _viewer_loop(self):
        while True:
            job, self._view = getattr(self, "_view", None), None
            if job is None:
                time.sleep(0.01)
                continue
            view = self.annotate(*job)
            if time.time() - getattr(self, "_live_t", 0) > LIVE_EVERY_S:
                # image « en direct » pour la page de suivi à distance (scripts/live_server.py)
                self._live_t = time.time()
                live = LIVE_DIR / "live.jpg"
                cv2.imwrite(str(live.with_suffix(".tmp.jpg")), view, [cv2.IMWRITE_JPEG_QUALITY, 80])   # pleine résolution
                try:
                    os.replace(live.with_suffix(".tmp.jpg"), live)
                except OSError:
                    pass
            cv2.imshow("Clash AI", cv2.resize(view, (int(view.shape[1] * 1.25), int(view.shape[0] * 1.25))))
            cv2.waitKey(1)

    def _emotes(self, dev, img, now):
        """Messages rapides pour faire plus humain : « Bonne chance » au début, « Bien joué » vers la fin.
        Positions dans data/emotes.json ; sans ce fichier, on ouvre le menu une fois et on en garde une capture
        (runs/emote_menu.jpg) pour régler les positions."""
        cfg_f = Path(__file__).resolve().parents[1] / "data/emotes.json"
        t = now - self._emote_t0
        if not cfg_f.exists():
            if "calib" not in self._emotes_done and t > 4:
                self._emotes_done.add("calib")
                dev.tap(*B.px(img, *EMOTE_CHAT))
                time.sleep(0.8)
                shot = dev.frame()[0]
                if shot is not None:
                    cv2.imwrite(str(LIVE_DIR / "emote_menu.jpg"), shot)
                dev.tap(*B.px(img, *EMOTE_CHAT))                  # referme le menu
            return
        cfg = json.loads(cfg_f.read_text(encoding="utf-8"))
        for key, when in (("start", 3.0), ("end", 170.0)):
            if key not in self._emotes_done and t > when and cfg.get(key):
                self._emotes_done.add(key)
                chat, pos = B.px(img, *cfg.get("chat", EMOTE_CHAT)), B.px(img, *cfg[key])

                def send(chat=chat, pos=pos):
                    dev.tap(*chat)
                    time.sleep(0.5)                      # dans un fil à part : l'IA continue de jouer (30/09 : pic à 578 ms)
                    dev.tap(*pos)
                threading.Thread(target=send, daemon=True).start()

    def _watch_spells(self, units, img, now):
        """Mesure le temps de vol de nos sorts : du tap jusqu'à ce que le détecteur voie l'effet près de la cible.
        Sert à caler SPELL_IMPACT_S (brain.py) sur de vrais matchs."""
        from clashai.brain import KING_Y, _tile_dist
        h, w = img.shape[:2]
        for sp in list(self.spells_pending):
            if now - sp["t_tap"] > 4:
                self.spells_pending.remove(sp)            # jamais vu : on abandonne (noté quand même)
                self.spells_log.append({"card": sp["card"], "t": round(sp["t_tap"], 2), "aimed": sp.get("reason", ""),
                                        "hit": None, "hit_value": None})
                continue
            for u in units:
                # l'EXPLOSION (le sort arrivé sur sa cible), pas la boule encore en vol à 4 cases : sinon on
                # mesurait autour de la boule pendant le vol, et un bon tir passait pour « raté »
                if u.name == sp["card"] and _tile_dist(u.center[0] / w, u.center[1] / h, sp["x"], sp["y"]) < 1.5:
                    # ce qui est VRAIMENT dans le rayon quand le sort tombe (vs ce qui était visé au tir)
                    from clashai import card_info
                    from clashai.opponent import SPELLS, UNIT2CARD
                    r = card_info.spell_radius(sp["card"]) + 0.5
                    ix, iy = sp["x"], sp["y"]                     # le point visé = centre de l'explosion
                    hit = [e.name for e in units if e.enemy and e.name != sp["card"] and e.name in UNIT2CARD
                           and e.name not in SPELLS                  # son Poison au sol n'est pas une cible
                           and _tile_dist(e.center[0] / w, e.center[1] / h, ix, iy) <= r]
                    value = sum(UNIT2CARD[n][1] / max(UNIT2CARD[n][2], 1) for n in hit)   # élixir par unité touchée
                    self.spells_log.append({"card": sp["card"], "flight_s": round(now - sp["t_tap"], 2),
                                            "dist_tiles": round(_tile_dist(sp["x"], sp["y"], 0.5, KING_Y), 1),
                                            "t": round(sp["t_tap"], 2), "aimed": sp.get("reason", ""),
                                            "hit": hit, "hit_value": round(value, 1)})
                    self.spells_pending.remove(sp)
                    break

    def _opp_update(self, units, now, frame_h):
        """Modèle de l'adversaire : cartes posées (journal) et carte lourde (punition), même pendant une pose."""
        new = self.opp.update(units, now, frame_h)
        for c in new:
            self.opp_log.append({"t": round(now, 2), "card": c, "elixir_after": round(self.opp.elixir, 1)})
        if any(COSTS.get(c, 0) >= 6 for c in new):
            self.brain.opp_heavy_t = now

    def _observe(self, img, info):
        """Pendant qu'une carte se pose : on continue de suivre les unités et d'afficher (pas de décision)."""
        now = time.time()
        units = self.det(img)
        self._watch_spells(units, img, now)
        self.match.update(img, units, now)
        self._opp_update(units, now, img.shape[0])
        self._show(img, units, None, info)

    def _learn_new_card(self, img, hand, ready):
        """Une carte du deck sans exemple (nouvelle dans le deck) : quand un emplacement
        en couleur reste illisible 3 images de suite, c'est elle -> on l'apprend."""
        from clashai.cards import DECK
        missing = [c for c in DECK if c not in H.known_cards()]
        if len(missing) != 1:
            return False
        streak = getattr(self, "_unknown_streak", {})
        learned = False
        for slot in range(4):
            crop = H.card_crop(img, slot)
            name, score = H.identify(crop)
            if hand[slot] is None and ready[slot] and score < 0.45:
                streak[slot] = streak.get(slot, 0) + 1
                if streak[slot] >= 3:
                    H.learn(crop, missing[0])
                    print(f"   carte apprise : {missing[0]} (score {score:.2f})", flush=True)
                    streak.clear()
                    learned = True
                    break
            else:
                streak[slot] = 0
        self._unknown_streak = streak
        return learned

    def _draw_opponent(self, v):
        from clashai.opponent import UNIT2CARD
        h, w = v.shape[:2]
        v[0:92] = (v[0:92] * 0.15).astype(v.dtype)
        o = self.opp
        costs = {c: cost for c, cost, _ in UNIT2CARD.values()}
        # main probable : 4 cases, comme nos cartes
        hand = (o.hand + ["?"] * 4)[:4] if len(o.deck) >= 5 else ["?"] * 4
        cw, x0 = (w - 120) // 4, 6
        cv2.putText(v, "MAIN ADVERSE", (x0, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 200, 255), 1, cv2.LINE_AA)
        for i, c in enumerate(hand):
            x = x0 + i * cw
            known = c != "?"
            cv2.rectangle(v, (x, 18), (x + cw - 6, 60), (60, 60, 160) if known else (60, 60, 60), -1)
            cv2.rectangle(v, (x, 18), (x + cw - 6, 60), (120, 120, 255) if known else (110, 110, 110), 1)
            cv2.putText(v, _ascii(c)[:11], (x + 4, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            if known and c in costs:
                cv2.circle(v, (x + cw - 18, 50), 8, (200, 60, 200), -1)
                cv2.putText(v, str(costs[c]), (x + cw - 22, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        # prochaine carte à revenir
        nx = w - 108
        cv2.putText(v, "SUIVANTE", (nx, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.rectangle(v, (nx, 18), (w - 6, 60), (70, 70, 70), -1)
        cv2.putText(v, _ascii(o.next_in or "?")[:12], (nx + 4, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (230, 230, 230), 1, cv2.LINE_AA)
        # barre d'élixir, rose comme la nôtre, graduée de 0 à 10
        bx0, bx1, by = 6, w - 6, 70
        cv2.rectangle(v, (bx0, by), (bx1, by + 14), (60, 30, 60), -1)
        fill = int((bx1 - bx0) * max(0.0, min(10.0, o.elixir)) / 10)
        cv2.rectangle(v, (bx0, by), (bx0 + fill, by + 14), (220, 80, 220), -1)
        for k in range(1, 10):
            x = bx0 + (bx1 - bx0) * k // 10
            cv2.line(v, (x, by), (x, by + 14), (40, 20, 40), 1)
        cv2.putText(v, f"elixir adverse ~{o.elixir:.1f}  ({len(o.deck)}/8 cartes vues)", (bx0 + 4, by + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    def annotate(self, img, units, d, info):
        v = img.copy()
        # bandeau du haut : l'adversaire comme notre bas d'écran (barre d'élixir + sa main probable)
        h, w = v.shape[:2]
        self._draw_opponent(v)
        draw(v, units, self.det.trails)
        if d:
            h, w = v.shape[:2]
            p = (int(d.x * w), int(d.y * h))
            cv2.circle(v, p, 16, (0, 255, 255), 3)
            cv2.arrowedLine(v, B.px(v, B.CARD_X[d.slot], B.CARD_Y - 0.05), p, (0, 255, 255), 2)
            info = [d.reason] + info
        for i, line in enumerate(info):
            line = _ascii(line)
            cv2.putText(v, line, (8, 110 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(v, line, (8, 110 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return v

    def play_battle(self, dev, game_id: str, params: dict | None = None) -> dict:
        """Joue un combat jusqu'au bout avec la variante de stratégie `params`."""
        self.brain = Brain(params)
        self.ctx = None                          # type du deck adverse, reconnu en cours de partie
        self.opp, self.opp_log = Opponent(), []
        self.spells_pending, self.spells_log = [], []
        self.perf, self.perf_log = {"det_ms": 0.0}, []
        self.match = MatchState()
        folder = os.path.join(self.out, game_id)
        os.makedirs(folder, exist_ok=True)
        self.det.tracker.reset() if self.det.tracker is not None else None
        log, last_play, gone, t_prev, n, refused = [], 0.0, None, time.time(), 0, 0
        self._emote_t0, self._emotes_done = time.time(), set()
        last_n, t_new = -1, time.time()
        while True:
            # une image NOUVELLE à chaque tour : sans ça, un flux figé ou coupé faisait tourner la boucle à 100 %
            # sur la même image, sans jamais finir le combat (et le suivi recevait des images en double)
            dev.wait_frame(timeout=0.5, after=last_n)
            img, t_recv, frame_n = dev.frame()
            if img is None or frame_n == last_n:
                if not getattr(dev, "alive", True) or time.time() - t_new > self.STALL_S:
                    raise StreamLost(f"plus d'image du téléphone depuis {time.time() - t_new:.0f} s")
                continue
            last_n, t_new = frame_n, time.time()
            t_loop = time.perf_counter()
            if not B.in_battle(img):
                gone = gone or time.time()
                if time.time() - gone > 5:
                    break
                time.sleep(0.2)
                continue
            gone = None
            now = time.time()
            self._emotes(dev, img, now)
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
                if ok and d.card not in ("arrows", "fireball"):     # nos sorts : note_our_spell ci-dessous
                    self.opp.note_our_troop(time.time(), d.card, d.x * w)
                if ok and d.card in ("arrows", "fireball"):
                    self.opp.note_our_spell(time.time(), d.x * w, d.y * h, card=d.card)
                    # vol mesuré depuis la DÉCISION : c'est ce délai que brain.SPELL_IMPACT_S doit prévoir
                    self.spells_pending.append({"card": d.card, "x": d.x, "y": d.y, "t_tap": now, "reason": d.reason})
                log.append({"t": round(now, 2), "card": d.card, "slot": d.slot, "x": round(d.x, 3), "y": round(d.y, 3), "tile": d.tile,
                            "reason": d.reason, "ok": ok, "play_ms": play_ms, "elixir": el, "hand": hand,
                            "opp_elixir": round(self.opp.elixir, 1), "trade": d.trade,
                            "units": [(u.name, u.enemy, u.center) for u in units],
                            # après les corrections de camp du cerveau (notre Géant lu « ennemi » ? -> visible ici)
                            "seen": [(s.name, s.enemy, round(s.x, 3), round(s.y, 3))
                                     for s in getattr(self.brain, "last_seen", [])]})
                if ok:
                    last_play = time.time()
                    self.brain.played(d, last_play)   # le cerveau ne retient que les cartes vraiment posées
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
                 str(sum(COSTS.get(c, 0) for c in self.opp.played))]
        for i, line in enumerate(lines):
            cv2.putText(card, _ascii(line), (10, 32 + 38 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(folder, "opponent_summary.jpg"), card)
        with open(os.path.join(folder, "opponent.json"), "w") as f:
            json.dump({"played": self.opp_log, "deck": self.opp.deck}, f, indent=1)
        return {"game": game_id, "played": n, "refused": refused, "params": self.brain.p,
                "trade_balance": self.brain.trade_balance,
                "enemy_deck": self.opp.deck, "ctx": self.ctx}
