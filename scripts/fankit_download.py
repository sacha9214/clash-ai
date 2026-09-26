"""Télécharge les personnages du Fan Kit officiel de Supercell (fankit.supercell.com, type « Characters »).

Rendus détourés (fond transparent) de toutes les cartes, y compris les récentes : héros, évolutions, nouvelles
cartes. Servent à fabriquer des images d'entraînement pour les cartes que le détecteur ne connaît pas.
Accord de Sacha (2026-09-26). 640 px de large : assez pour des unités qui font 40-120 px à l'écran.

  .venv\\Scripts\\python scripts/fankit_download.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import requests

API = "https://fankit.supercell.com/api/assets/search/345"
OUT = Path("D:/clash-ai-dataset/fankit/characters")
WIDTH = 640


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta, page = [], 1
    while True:
        j = requests.get(API, params={"limit": 100, "page": page, "order": "NEWEST", "asset-type16": "Characters"},
                         timeout=60).json()
        meta += [{k: a.get(k) for k in ("id", "title", "created", "width", "height", "generic_url", "ext")} for a in j["data"]]
        if not j.get("hasMore"):
            break
        page += 1
    (OUT.parent / "characters.json").write_text(json.dumps(meta, indent=0), encoding="utf-8")
    print(f"{len(meta)} personnages", flush=True)
    ok = 0
    for i, a in enumerate(meta, 1):
        f = OUT / f"{a['id']}_{a['title']}.png"
        if f.exists():
            ok += 1
            continue
        url = a["generic_url"].replace("{width}", str(min(WIDTH, a["width"] or WIDTH)))
        for attempt in range(3):
            try:
                r = requests.get(url, timeout=60)
                if r.ok:
                    f.write_bytes(r.content)
                    ok += 1
                    break
            except requests.RequestException:
                time.sleep(5)
        time.sleep(0.2)                            # rythme poli pour le site
        if i % 100 == 0:
            print(f"{i}/{len(meta)}", flush=True)
    size = sum(p.stat().st_size for p in OUT.glob("*.png")) / 1e6
    print(f"terminé : {ok}/{len(meta)} images, {size:.0f} Mo", flush=True)


if __name__ == "__main__":
    main()
