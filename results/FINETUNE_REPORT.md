# Réglage fin du détecteur

| Modèle | Précision | Rappel | F1 | Camp | ms/image |
|---|---|---|---|---|---|
| Avant (YOLO11 TensorRT) | 97.5% | 90.5% | **0.939** | 100.0% | 3.4 |
| Après réglage fin | 98.0% | 89.0% | **0.933** | 99.9% | 3.3 |

**Pas meilleur : l’IA garde le modèle actuel.**

Unités les plus ratées après : skeleton (330), royal-recruit (192), zappy (150), royal-hog (142), elite-barbarian (135), dirt (98), king-tower (96), musketeer (96), ice-golem (93), lumberjack (82)