#!/bin/zsh
# Supervise la nuit : lots de 5 combats (avec apprentissage des stratégies), puis
# améliorations cartes/tours, et on recommence. `touch runs/games/STOP` pour arrêter proprement.
cd "$(dirname "$0")/.."
PY=.venv-katacr/bin/python
fails=0
while [ ! -f runs/games/STOP ]; do
  $PY scripts/autoplay.py --games 5 >> runs/night.log 2>&1
  if tail -3 runs/night.log | grep -q "écran inconnu"; then
    fails=$((fails+1)); echo "$(date) écran inconnu ($fails)" >> runs/night.log
    [ $fails -ge 3 ] && { echo "$(date) arrêt : 3 écrans inconnus de suite" >> runs/night.log; break; }
    sleep 60; continue
  fi
  fails=0
  $PY scripts/upgrade.py >> runs/night.log 2>&1
done
echo "$(date) nuit terminée" >> runs/night.log
