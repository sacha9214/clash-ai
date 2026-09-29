r"""Enchaîne de vrais combats (Ladder) toute la nuit et tient un journal.

États reconnus : accueil (bouton Battle), recherche d'adversaire, combat, écran de fin (OK).
Tout écran inconnu qui dure -> capture + arrêt, pour ne jamais cliquer à l'aveugle
(boutique, offres payantes…).

  .venv-yolo\Scripts\python scripts/autoplay.py --games 10
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import av  # noqa: F401,E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from clashai import battle as B  # noqa: E402
from clashai.agent import Agent, StreamLost  # noqa: E402
from clashai.device import Device  # noqa: E402
from clashai import strategy as ST  # noqa: E402

BATTLE_BTN = (0.50, 0.822)       # bouton jaune « Battle » de l'accueil = même endroit que « OK » en fin de combat
OUR_START_S = 90                 # combat commencé moins de 90 s après NOTRE tap sur Battle (recherche : 45 s max)
END_WAIT_S = 15                  # l'écran de fin (bouton OK) doit apparaître dans ce délai pour lire le score


def color_at(img, fx, fy, r=6):
    x, y = B.px(img, fx, fy)
    return img[y - r:y + r, x - r:x + r].reshape(-1, 3).mean(0)   # BGR


def is_home(img):
    b, g, r = color_at(img, 0.372, 0.797)       # coin du bouton Battle (hors texte)
    return r > 120 and g > 80 and b < 70 and r > g   # jaune (même assombri par un tutoriel)


def is_blue_btn(img, fx, fy):
    b, g, r = color_at(img, fx, fy)
    return b > 200 and g > 120 and r < 120


def end_ok(img):
    """Position du bouton OK bleu de fin de combat (seul au centre, ou à droite de « Play Again »)."""
    for fx, fy in ((0.40, 0.828), (0.60, 0.828)):
        if is_blue_btn(img, fx, fy):
            return (0.5 if fx == 0.40 else 0.655), 0.825
    return None


def is_end(img):
    return end_ok(img) is not None


def is_chest(img):
    """Écran d'ouverture de coffre : fond uni et coloré (orange, bleu, violet…) tout
    en haut, sans l'interface de l'accueil ni l'herbe de l'arène."""
    pts = np.array([color_at(img, fx, 0.04, r=8) for fx in (0.08, 0.5, 0.92)])
    uniform = np.abs(pts - pts.mean(0)).max() < 25
    sat = (pts.max(1) - pts.min(1)).mean()
    return bool(uniform and sat > 60)


def result_of(img):
    """Compare les couronnes dorées : les nôtres (bandeau bleu, bas) et les siennes (bandeau rouge, haut).
    On compte des PIXELS (mêmes icônes des deux côtés) : les deux bandeaux n'ont pas la même hauteur, comparer des
    proportions donnait « victoire » à chaque égalité 1-1, 2-2 ou 3-3."""
    h, w = img.shape[:2]

    def gold(y0, y1):
        band = img[int(y0 * h):int(y1 * h), int(0.12 * w):int(0.88 * w)].reshape(-1, 3).astype(int)
        b, g, r = band[:, 0], band[:, 1], band[:, 2]
        return int(((r > 200) & (g > 140) & (b < 90) & (r - b > 130)).sum())

    ours, theirs = gold(0.44, 0.55), gold(0.17, 0.32)
    area = (int(0.55 * h) - int(0.44 * h)) * (int(0.88 * w) - int(0.12 * w))
    if max(ours, theirs) < 0.01 * area or abs(ours - theirs) < 0.25 * max(ours, theirs):
        return "draw"                   # aucune couronne, ou autant de chaque côté
    return "win" if ours > theirs else "loss"


def wait_end_screen(dev, timeout=END_WAIT_S):
    """Image de l'écran de fin (bouton OK visible), ou None s'il n'apparaît pas : pas de score lu au hasard."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        img, _, _ = dev.frame()
        if img is not None and is_end(img):
            time.sleep(0.8)                      # couronnes animées : on laisse l'écran se poser
            return dev.frame()[0]
        time.sleep(0.3)
    return None


def back() -> None:
    """Touche Retour d'Android : ferme la Route des trophées et les menus ouverts par erreur."""
    import subprocess
    from clashai.device import ADB
    subprocess.run([ADB, "shell", "input", "keyevent", "KEYCODE_BACK"], timeout=10,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=5)
    ap.add_argument("--out", default="runs/games")
    ap.add_argument("--show", action="store_true", help="fenêtre en direct sur le Mac")
    a = ap.parse_args()
    agent = Agent(a.out, show=a.show)
    journal = os.path.join(a.out, "journal.jsonl")

    with Device() as dev:
        played, unknown_since, chest_until, battle_tap = 0, None, 0.0, None
        while True:   # on ne sort qu'à l'accueil (après coffres et récompenses du dernier combat)
            img, _, _ = dev.frame()
            if B.in_battle(img) and played < a.games:
                gid = time.strftime("%Y%m%d-%H%M%S")
                # un combat déjà en cours au lancement du script (ou repris après une pause) : joué, mais son résultat
                # ne dit rien de la stratégie (elle n'a joué que la fin, élixir et chrono faux)
                ours = battle_tap is not None and time.time() - battle_tap < OUR_START_S
                battle_tap = None
                print(f"[{gid}] combat {played + 1}/{a.games}" + ("" if ours else " (pris en cours : non compté)"), flush=True)
                stats = ST.load()
                params = ST.choose(stats)
                try:
                    summary = agent.play_battle(dev, gid, params)
                except StreamLost as e:
                    print(f"flux vidéo perdu : {e} -> arrêt", flush=True)
                    break
                end = wait_end_screen(dev)
                summary["result"] = result_of(end) if end is not None else "unknown"
                summary["counted"] = ours and end is not None
                cv2.imwrite(os.path.join(a.out, gid, "end.jpg"), end if end is not None else dev.frame()[0])
                if summary["counted"]:
                    ST.record(stats, summary["params"], summary["result"])   # paramètres réellement joués
                    ctx = summary.get("ctx") or ST.archetype(summary.get("enemy_deck") or [])
                    ST.record_ctx(ST.load_ctx(), ctx, summary["params"], summary["result"])
                with open(journal, "a") as f:
                    f.write(json.dumps(summary) + "\n")
                print("   ", summary, flush=True)
                played += 1
                unknown_since = None
            elif is_end(img):
                dev.tap(*B.px(img, *end_ok(img)))
                time.sleep(3)
                # Juste après un combat : écrans d'ouverture de coffre / récompenses.
                # On tape au centre (seulement dans ce contexte, borné) jusqu'au retour à l'accueil.
                for _ in range(25):
                    img, _, _ = dev.frame()
                    if is_home(img) or B.in_battle(img):
                        break
                    dev.tap(*B.px(img, 0.5, 0.42))
                    time.sleep(1.3)
            elif is_home(img):
                if played >= a.games or os.path.exists(os.path.join(a.out, "STOP")):
                    break
                dev.tap(*B.px(img, *BATTLE_BTN))
                battle_tap = time.time()
                time.sleep(4)
                unknown_since = None
            elif is_chest(img) or (chest_until and time.time() < chest_until):
                if is_chest(img):
                    chest_until = time.time() + 8
                # alterne : centre (coffre, carte) et bouton OK des écrans « niveau supérieur »
                taps = getattr(dev, "_reward_taps", 0)
                dev._reward_taps = taps + 1
                # + le bouton OK tout en bas (nouvelle arène : écran pris pour un coffre, bloqué 50 min le 27/09)
                if taps % 4 == 3:
                    back()          # Route des trophées : les taps ne la ferment pas, la touche Retour oui (29/09 : 6 min bloqué)
                else:
                    dev.tap(*B.px(img, 0.5, (0.42, 0.66, 0.966)[taps % 4]))
                time.sleep(1.3)
                unknown_since = None
            else:
                unknown_since = unknown_since or time.time()
                # Route des trophées / nouvelle arène : bouton OK tout en bas. Sans risque ailleurs
                # (sur l'accueil c'est la barre d'onglets). On l'essaie toutes les 10 s.
                waited = time.time() - unknown_since
                if waited > 10 and int(waited) % 10 == 0:
                    dev.tap(*B.px(img, 0.5, 0.966))
                    if waited > 45 and int(waited) % 20 == 0:
                        # écran inconnu (Route des trophées, menu ouvert…) : Retour le ferme. Pas avant 45 s : ce peut
                        # être la recherche d'adversaire, que Retour annulerait
                        back()
                    time.sleep(1.0)
                if time.time() - unknown_since > 45:   # matchmaking dure rarement plus
                    cv2.imwrite(os.path.join(a.out, "unknown.jpg"), img)
                    print("écran inconnu depuis 45 s : arrêt (voir unknown.jpg)", flush=True)
                    break
                time.sleep(0.5)
    print("fin")


if __name__ == "__main__":
    main()
