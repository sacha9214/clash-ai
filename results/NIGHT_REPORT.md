# Nuit du détecteur

Entraînement : 8.07 h. Test : séquences vidéo entières jamais vues.

| Modèle | Précision | Rappel | F1 | Camp | ms/image |
|---|---|---|---|---|---|
| KataCR (actuel, 2 modèles) | 97.0% | 86.8% | **0.916** | 99.4% | 28.4 |
| YOLO11 (nouveau, .pt) | 97.4% | 90.3% | **0.937** | 100.0% | 10.2 |
| YOLO11 TensorRT (FP16) | — | — | — | — | — |

**Verdict : le nouveau détecteur est MEILLEUR que KataCR** (F1 0.937 contre 0.916, 10.2 ms contre 28.4 ms).

Unités les plus ratées (nouveau) : skeleton (243), elite-barbarian (162), royal-recruit (158), zappy (120), royal-hog (120), king-tower (97), ice-spirit (94), lumberjack (93)

Unités les plus ratées (KataCR) : skeleton (407), royal-recruit (378), royal-hog (197), zappy (177), rage (156), elite-barbarian (134), ice-golem (132), musketeer (128)

Rien n'a été changé dans l'IA : le passage au nouveau détecteur attend le feu vert de Sacha.