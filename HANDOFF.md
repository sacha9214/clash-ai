# Reprise du projet

Colle ce fichier à Claude Code pour reprendre.

## Où on en est (2026-09-25, soir, PC Windows RTX 5070)
- IA Clash Royale qui joue sur le téléphone de Sacha (REDMAGIC 11 Air, USB + scrcpy), compte secondaire « claude ».
- ~49 matchs Ladder, Arène 2. Deck : Géant, Chevalier, Mini P.E.K.K.A, Archères, Valkyrie, Canon, Flèches, Boule de feu.
- Cerveau à règles (`clashai/brain.py`) + apprentissage des stratégies (bandit, `clashai/strategy.py`, état dans `learning/`).
- Suivi adverse (`clashai/opponent.py`), fenêtre en direct : `scripts/autoplay.py --games 1 --show`.

### Fait sur le PC
- Windows : `device.py` trouve adb / scrcpy-server tout seul (env `ADB`, `SCRCPY_SERVER_PATH` sinon) ; torch 2.11+cu128
  dans `.venv-katacr` (sm_120) ; KataCR détecte en 35 ms/image sur le GPU (110 ms sur le Mac).
  torch >= 2.6 refuse les .pt d'ultralytics 8.1 -> `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` (detect.py, train_detector.py).
- Norton Antivirus inspecte le HTTPS : `UV_SYSTEM_CERTS=1` (variable utilisateur) et `pip-system-certs` dans `.venv`.
  Python via uv dans `%LOCALAPPDATA%\uv\python` (`UV_PYTHON_INSTALL_DIR`).
- Grille 18x32 cases (`clashai/tiles.py`) : calibrée sur l'écran du téléphone et sur l'arène 568x896 des vidéos.
  Les troupes sont posées au centre d'une case ; chaque coup note sa case.
- 13 règles tirées de 5 vidéos d'explication (transcrites avec faster-whisper, `scripts/transcribe.py`) :
  directes (échange d'élixir, ignorer les petites unités seules, recalage élixir adverse après une carte lourde,
  archères séparées, défense au centre contre la mêlée mono-cible, canon à 4 cases, Valkyrie contre Tonneau,
  garder le contre de leur grosse menace) et options testées par le bandit (`punish_opposite`,
  `giant_when_counter_out`, `fireball_patient`, `giant_spot=corner`, `cycle_at` 7-8).
- Sorts : jamais sur le Roi ennemi (l'activerait) ; visée au point d'impact (temps de vol depuis notre Roi)
  et on ne compte que les unités encore dans le cercle à l'impact. Distances en cases.
- Plus de gel pendant la pose d'une carte : la boucle continue de suivre/afficher, captures en arrière-plan,
  durée de chaque coup dans `decisions.jsonl` (`play_ms`, médiane 160 ms).
- Vidéos YouTube (sur `D:\clash-ai-videos`, pas dans le dépôt) : 8 vidéos de gameplay Géant en 1080p + 5 d'explication.
  `scripts/extract_videos.py` localise l'arène (par les tours), analyse 5 images/s (x6 le temps réel) et écrit
  chaque carte jouée (camp, instant, case, plateau) dans `runs/videos/`. ~1 770 coups extraits.

## 26/09 (en cours, autonome)
- Détecteur de l'IA : NOTRE YOLO11s TensorRT (`clashai/detect_yolo.py`, `models/yolo/`), .venv-yolo.
  Test (séquences jamais vues) : F1 0.939 vs 0.916 KataCR, 8.5 ms vs 28 ms. KataCR en secours (CLASHAI_DETECTOR=katacr).
- Placement appris des pros (`clashai/placement.py`, `scripts/train_placement.py`) : 4.3 cases d'écart vs 5.1 pour
  les règles (vidéos jamais vues). Règles : côté de l'ennemi le plus avancé, Canon à l'avance / contre tout le sol.
- Nouvelles cartes / héros / évolutions : rendus du Fan Kit officiel (`scripts/fankit_*.py`, D:\clash-ai-datasetankit),
  quelques boîtes réelles (label_tool.py, bug d'écrasement corrigé), pseudo-étiquetage après v2.
- Tout passe par `scripts/pipeline.py` (processus Windows, journal `runs/pipeline.log`) : téléchargement jusqu'à
  50 h (arrêté à 53 h) + analyse avec NOTRE YOLO11 TensorRT (6 vidéos en parallèle, CPU saturé) -> comparaison + modèle de placement
  -> réglage fin du détecteur -> v2 (nouvelles cartes) -> pseudo-étiquetage ; publie code, rapports (`results/`)
  et petits modèles sur GitHub après chaque étape.
- Pas de téléphone le 26/09 : aucun match ; tout reste à valider en match ensuite.

## 27/09 : grande revue (sans téléphone : tout est validé par tests et par rejeu des 52 matchs, PAS en match)
- Tests : `python -m pytest -q tests` (cerveau, adversaire, bandit, détecteurs, boucle ; tout hors téléphone).
  Rejeu des décisions enregistrées avec le cerveau actuel : `python scripts/replay_logs.py --games <runs/games>`.
- Cerveau :
  - Sorts jugés sur l'élixir détruit (`spell_value`), jamais sur le Roi ennemi.
  - Mini P.E.K.K.A (pas les Archères) contre un Géant.
  - Placements précis (`Decision.precise`) jamais déplacés par le modèle.
  - `Brain.played()` n'est appelé qu'après un tap vérifié.
  - Troupes virtuelles mortelles, camps corrigés, élixir jamais perdu à 10.
- Stats des cartes ramenées au même niveau (rareté). Vitesse des unités horodatée (`clashai/motion.py`).
- Bandit factorisé (`strategy.param_table()` pour voir ce qu'il a appris). Anciens matchs = fonctions absentes (`LEGACY`).
- Boucle et autoplay :
  - Flux coupé -> arrêt propre (`StreamLost`).
  - Égalités lues « nul » ; combats pris en cours non comptés ; résultat lu seulement sur l'écran de fin.
  - Tours Canonnier/Duchesse reconnues.
- À valider en match d'abord : les sorts (seuil 0.8 x coût), le camp corrigé des unités, le modèle adverse.

### Leçons (erreurs trouvées, à ne pas refaire)
1. Un paramètre du bandit peut cacher un comportement nuisible : `fireball_min=1` (36 matchs sur 52) lançait la Boule
   de feu sur un Géant seul dès qu'un tireur existait AILLEURS sur le terrain. Vérifier ce que fait chaque variante
   dans les journaux (`decisions.jsonl`), pas seulement son taux de victoire.
2. Un seuil lu sur l'écran doit être atteignable : l'élixir se lit au plus ~9.5 -> `giant_elixir` 10 puis `cycle_at` 10
   n'arrivaient jamais. Toujours comparer un seuil au maximum réellement mesuré.
3. Les valeurs « niveau 1 » des fichiers du jeu sont au 1er niveau de CHAQUE rareté (rare = niveau 3, épique = 6) :
   comparer des cartes demande de les ramener au même niveau.
4. Une fonction de décision ne doit rien retenir : la boucle redécide à chaque image et ne joue qu'une décision sur
   plusieurs (délai entre deux cartes). Mémoriser seulement ce qui est vraiment joué.
5. Contre une unité qui ne vise que les bâtiments, « gagner le duel » est trivial : le vrai critère est le temps
   avant qu'elle n'abîme la tour.
6. Vitesse = positions HORODATÉES : le détecteur tourne aussi sur les rafales d'images pendant la pose d'une carte.
7. Comparer des proportions de zones de tailles différentes fausse la comparaison (couronnes : égalités -> « victoire »).
8. Un test doit d'abord ÉCHOUER sur l'ancien code : un test de tours avec une horloge factice (0-9 s) passait sur le
   code bogué, parce que l'état initial utilise l'horloge réelle.
9. Tout filtre « crédible / pas crédible » peut aveugler le cerveau (unités invoquées, évolutions, deck verrouillé
   par des fantômes) : ne cacher une unité que si c'est vérifiable.
10. L'avance d'élixir se compte main + TERRAIN : sans les troupes posées, l'IA se croyait « en retard » juste après
    son Géant et ne le soutenait plus (135 soutiens supprimés au rejeu).
11. Un bilan d'échanges doit créditer chaque attaque adverse UNE fois : chaque carte posée contre le même Géant
    « gagnait » 5 élixir (+371 fictifs sur 52 matchs).
- Avance d'élixir (`edge_push`, bandit) : Géant plus tôt avec 3+ d'avance, pas d'attaque avec 3+ de retard.
  Bilan des échanges estimé : à l'écran, dans `decisions.jsonl` (`trade`, `opp_elixir`) et `journal.jsonl`
  (`trade_balance`) -> comparer bilan et victoires quand il y aura des matchs.

## Nuit du 25 au 26/09 (autonome, hors de Claude)
- `scripts/night_detector.py` (PID 39148) : YOLO11s, 64 199 images (60 000 synthétiques + 4 199 réelles), 8 h max,
  puis note sur le test (séquences jamais vues ; KataCR : F1 0.916), export TensorRT et rapport
  `runs/detector/NIGHT_REPORT.md` (journal : `runs/detector/night.log`).
- `scripts/night_videos.py` : 64 vidéos récentes et variées (30 h, 22 recherches : decks, bas niveau, top ladder)
  -> `D:\clash-ai-videos80p` (+ `manifest.jsonl`), analysées APRÈS l'entraînement (GPU plein),
  puis comparaison des placements -> `runs/videos/placement_report.txt` (journal : `runs/videos/night_videos.log`).
- `scripts/night_matches.py` : prêt (30 combats automatiques) mais ANNULÉ cette nuit : pas de téléphone le 26/09.
- Rien n'est changé dans l'IA : le passage au nouveau détecteur attend le feu vert de Sacha.

## À faire
1. Vidéos : découper les combats de façon fiable (chrono 3:00 plutôt que « tours visibles »), filtrer les cartes
   mal détectées par le deck vu, retrouver le vainqueur de chaque combat ; grille décalée d'~1 case sur les
   vieilles vidéos (graphismes 2017) -> caler sur la rivière.
2. Mesurer en match la vitesse réelle des sorts (`SPELL_SPEED_TILES` dans brain.py = estimations).
3. Savoir si le Roi ennemi est déjà actif (alors les sorts peuvent le toucher).
4. Entraîner le détecteur rapide : `scripts/train_detector.py` (GPU), nécessaire pour passer à des centaines d'heures de vidéo.
5. Imitation : apprendre des coups des gagnants dans les vidéos.
