"""Récupère les récompenses gratuites de la Route des trophées (bulle « Claim reward »).

Boucle : bulle -> route -> bouton vert (Collect / Choose) -> écrans de récompense ->
OK -> accueil ; tant qu'il y a un bouton vert. Aucune autre zone n'est touchée.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import av  # noqa: F401,E402
import cv2  # noqa: E402

from clashai.device import Device  # noqa: E402

exec(open(os.path.join(os.path.dirname(__file__), "upgrade.py")).read().split("with Device()")[0])

BUBBLE, ROAD_OK, CHOOSE_LEFT = (372, 680), (290, 1237), (176, 320)
claimed = 0
with Device() as d:
    for _ in range(12):
        d.tap(*BUBBLE)
        img = shot(d, "claim_road", wait=1.8)
        btns = green_buttons(img, 150, 1150)
        for _ in range(5):                # la récompense à prendre est souvent plus bas sur la route
            if btns:
                break
            d.drag(290, 1000, 290, 500, duration=0.4)
            img = shot(d, "claim_road", wait=1.2)
            btns = green_buttons(img, 150, 1150)
        if not btns:
            d.tap(*ROAD_OK)
            shot(d)
            break
        d.tap(*btns[-1])                  # le plus bas = la récompense la plus ancienne
        img = shot(d, "claim_1", wait=1.8)
        # « Choose your reward » : deux cartes en haut -> on prend la première
        if not green_buttons(img, 150, 1150) and not on_deck(img):
            d.tap(*CHOOSE_LEFT)
            shot(d, wait=1.5)
        for _ in range(4):                # écrans de récompense / niveau du Roi
            d.tap(290, 900)
            shot(d, wait=1.0)
        d.tap(*ROAD_OK)                   # si on est resté sur la route
        shot(d, wait=1.2)
        d.tap(326, 1222)                  # onglet Battle (sans effet si déjà à l'accueil)
        shot(d, wait=1.2)
        claimed += 1
print(f"récompenses récupérées : {claimed}", flush=True)
