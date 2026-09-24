"""Améliore tout ce qui peut l'être dans le deck (8 cartes + troupe de tour), avec l'or.

Sécurité : on ne confirme que si le bouton de confirmation montre une pièce d'or
(jaune) — jamais de gemmes. Chaque étape est vérifiée à l'image ; au moindre doute
on ferme et on passe à la suivante. Revient à l'écran d'accueil à la fin.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import av  # noqa: F401,E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from clashai.device import Device  # noqa: E402

W, H = 578, 1280
DECK = [(x, y) for y in (330, 555) for x in (78, 216, 356, 496)]
TOWER = (514, 890)
NAV_CARDS, NAV_BATTLE = (142, 1222), (326, 1222)   # l'onglet Battle se décale quand Cartes est ouvert
DETAIL_UPGRADE, CONFIRM = (290, 1004), (290, 884)
OUT = "runs/upgrades"
os.makedirs(OUT, exist_ok=True)


def region(img, x, y, rx, ry):
    sx, sy = img.shape[1] / W, img.shape[0] / H
    return img[int((y - ry) * sy):int((y + ry) * sy), int((x - rx) * sx):int((x + rx) * sx)].reshape(-1, 3).astype(int)


def green_frac(img, x, y, rx=35, ry=5):
    p = region(img, x, y, rx, ry)
    b, g, r = p[:, 0], p[:, 1], p[:, 2]
    return ((g > 120) & (g - r > 60) & (g - b > 50)).mean()


def gold_coin(img, x, y):
    """Y a-t-il une pièce d'or (jaune/orangé) sur le bouton ? (et pas une gemme verte/violette)"""
    p = region(img, x + 16, y + 14, 8, 8)
    b, g, r = p[:, 0], p[:, 1], p[:, 2]
    # or = orangé (R > G > B) ; une gemme serait verte (G > R) ou violette (B élevé)
    return ((r > 100) & (r > g) & (g > b + 30) & (r - b > 70)).mean() > 0.3


def green_buttons(img, y0=600, y1=1200):
    """Boutons verts visibles (centres, en coordonnées 578x1280), du plus haut au plus bas."""
    im = cv2.resize(img, (W, H))[y0:y1].astype(int)
    b, g, r = im[..., 0], im[..., 1], im[..., 2]
    mask = ((g > 120) & (g - r > 60) & (g - b > 50)).astype(np.uint8)
    n, _, st, cen = cv2.connectedComponentsWithStats(mask)
    out = [(int(cen[i][0]), int(cen[i][1]) + y0) for i in range(1, n)
           if st[i][4] > 1200 and st[i][2] > 60 and st[i][3] > 25]
    return sorted(out, key=lambda c: c[1])


def shot(d, name=None, wait=1.3):
    time.sleep(wait)
    img, _, _ = d.frame()
    if name:
        cv2.imwrite(f"{OUT}/{name}.jpg", cv2.resize(img, (W // 2, H // 2)))
    return img


def close(d):
    d.tap(290, 150)            # zone neutre du haut : ferme popups et fiches
    shot(d)


def try_upgrade(d, card_xy, label):
    x, y = card_xy
    img = shot(d, wait=0.3)
    bar_y = y + 94 if card_xy != TOWER else y + 84
    if green_frac(img, x, bar_y) < 0.3:
        return False                                  # pas assez de cartes
    d.tap(x, y)
    img = shot(d, f"{label}_1")
    btn = (x, y + (110 if card_xy == TOWER else 122))
    if green_frac(img, *btn, rx=30, ry=12) < 0.1:     # pas de bouton « Upgrade » vert dans la bulle
        close(d)
        return False
    d.tap(*btn)
    img = shot(d, f"{label}_2")
    btns = green_buttons(img)
    if not btns:
        close(d)
        return False
    detail = btns[-1]                                 # le bouton « Upgrade » de la fiche (en bas)
    d.tap(*detail)
    img = shot(d, f"{label}_3")
    above = [b for b in green_buttons(img) if b[1] < detail[1] - 40]   # « Confirm » apparaît au-dessus
    if not above or not gold_coin(img, *above[-1]):
        print(f"  {label} : confirmation absente ou pas en or -> annulé", flush=True)
        close(d); close(d)
        return False
    d.tap(*above[-1])
    for _ in range(4):                                # animations d'amélioration
        shot(d, wait=1.0)
        d.tap(290, 300)
    shot(d, f"{label}_4", wait=1.0)
    close(d)
    print(f"  amélioré : {label}", flush=True)
    return True


with Device() as d:
    d.tap(*NAV_CARDS)
    shot(d, "deck")
    done = 0
    for i, xy in enumerate(DECK):
        for _ in range(3):                            # plusieurs niveaux d'affilée si possible
            if not try_upgrade(d, xy, f"card{i}"):
                break
            done += 1
    while try_upgrade(d, TOWER, "tower"):
        done += 1
    for _ in range(2):                                # retour à l'accueil (2 fois : une bulle peut rester ouverte)
        d.tap(*NAV_BATTLE)
        shot(d, wait=1.2)
    shot(d, "home")
    print(f"améliorations : {done}", flush=True)
