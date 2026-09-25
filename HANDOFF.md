# Reprise sur le PC Windows (RTX 5070)

Colle ce fichier à Claude Code sur le PC pour reprendre.

## Où on en est (2026-09-25)
- IA Clash Royale qui joue sur le téléphone de Sacha (REDMAGIC 11 Air, USB + scrcpy), compte secondaire « claude ».
- ~46 matchs Ladder, ~30 victoires, Arène 2 (~500 trophées). Deck : Géant, Chevalier, Mini P.E.K.K.A, Archères, Valkyrie, Canon, Flèches, Boule de feu.
- Cerveau à règles (`clashai/brain.py`) + apprentissage des stratégies (bandit, `clashai/strategy.py`, état dans `learning/`).
- Suivi adverse (`clashai/opponent.py`) : élixir = 5 + gagné/s − coût des cartes vues ; deck, main probable, prochaine carte ; bandeau en haut.
- Boucle de nuit : `scripts/night.sh` (5 matchs → récompenses route des trophées → améliorations en or uniquement).
- Fenêtre en direct : `scripts/autoplay.py --games 1 --show`.

## À faire sur le PC
1. `git clone https://github.com/sacha9214/clash-ai && cd clash-ai && ./setup.sh` (Git Bash).
2. PyTorch pour RTX 50xx : `uv pip install --python .venv-katacr torch torchvision --index-url https://download.pytorch.org/whl/cu128`, vérifier que KataCR (ultralytics 8.1) tourne dessus.
3. Adapter `SERVER_LOCAL` dans `clashai/device.py` (chemin de scrcpy-server sous Windows), installer scrcpy + adb.
4. Brancher le téléphone sur le PC (câble/port : 3 déconnexions sur Mac).
5. Entraîner le détecteur rapide : `scripts/train_detector.py` (échec sur Mac, trop lent).
6. Idée validée : apprendre depuis des vidéos YouTube de gameplay (deck proche), téléchargement à confirmer avec Sacha.
