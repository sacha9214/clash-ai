# clash-ai

[![tests](https://github.com/sacha9214/clash-ai/actions/workflows/tests.yml/badge.svg)](https://github.com/sacha9214/clash-ai/actions/workflows/tests.yml)

**An AI that plays Clash Royale on a real Android phone — it sees the screen, reads the battle, decides like a player and learns which strategies win.**

No emulator, no game hacking: the AI watches the phone's video stream over USB (scrcpy), detects every unit with its own YOLO detector, reasons about elixir, trades and tower HP, taps its cards, and after every ladder match a bandit updates which play style wins.

> Automating Clash Royale is against Supercell's Terms of Service. Use a secondary account, friendly battles and Training Camp only.

## What it does

| | |
|---|---|
| 👀 **Sees** | Our own YOLO11s detector (TensorRT, **8.5 ms/frame** on an RTX 5070, F1 0.939 on unseen footage) + ByteTrack. Each unit's side (blue/red) and name are voted over its whole life, using where it was deployed and its health-bar colour; speeds come from timestamped positions. |
| 🧠 **Thinks** | Rule-based brain grounded in real card stats: picks the cheapest defender that *wins* a simulated duel, casts spells only when they destroy enough elixir (never a Fireball on a lone Giant), tracks the elixir advantage (hand + board), finishes weak towers with spells, counter-pushes behind surviving defenders, and adapts the last 30 s to the crown score. |
| 🕵️ **Reads the opponent** | Estimates his elixir, learns his deck and cycle (probable hand, next card), keeps a counter for his win condition, punishes heavy plays. |
| 🎯 **Places** | Where to drop a card comes from a model trained on ~14,600 moves of pro players (YouTube videos), except for precise tactical placements (Cannon pull, bridge punish). |
| 📈 **Learns** | A factorized Thompson-sampling bandit tries strategy settings (Giant timing, spell thresholds, elixir-advantage push…) across matches and keeps the winners — every game informs every parameter. |
| ✅ **Is tested** | 129 tests run without a phone or GPU (CI on every pull request), plus a replay of real match logs to measure every brain change. |

**Current deck:** Giant, Knight, Mini P.E.K.K.A, Valkyrie, Musketeer, Minions, Cannon, Fireball (`clashai/cards.py`).

## How it works

```
phone ──USB/scrcpy──▶ device.py ──frame──▶ detect_yolo.py (+ identity.py, motion.py)
                                              │ units: name, side, box, speed
                          battle.py / hand.py │ elixir, hand, playable cards
                                              ▼
            opponent.py (his elixir/deck) ─▶ brain.py ◀─ strategy.py (bandit settings)
            towers.py (HP, crowns, phase)      │ Decision: card, tile, reason
                                              ▼
                                  actions.py (tap, verified) ──▶ phone
```

| File | Role |
|---|---|
| `clashai/device.py` | Starts the scrcpy server on the phone, decodes H.264 in Python (low delay), injects touches (~83 ms tap-to-screen). |
| `clashai/detect_yolo.py` | Our detector + tracking; `identity.py` votes side/name per track, `motion.py` measures speed. `detect_katacr.py` is the fallback ([KataCR](https://github.com/wty-yy/KataCR), MIT). |
| `clashai/battle.py`, `hand.py` | Elixir bar, cards in hand, which ones are playable. |
| `clashai/brain.py` | Decisions: defense, spells, attack, support, counter-push, endgame. Every decision carries a human-readable reason. |
| `clashai/opponent.py` | Opponent model: elixir, deck (evidence-based), cycle, probable hand. |
| `clashai/towers.py` | Our towers' HP, enemy towers' HP (self-calibrated), crowns, match phase. |
| `clashai/card_info.py` | Card stats (range, speed, HP, damage…) normalized to the same level. |
| `clashai/placement.py` | Learned placement model (imitation of pro players). |
| `clashai/strategy.py` | The bandit that learns winning settings. |
| `clashai/agent.py` | The battle loop: see → think → play → log. |

## Quick start

Tested on Windows + RTX 5070 and macOS (M3). Needs a phone with USB debugging, [scrcpy](https://github.com/Genymobile/scrcpy) and `adb`.

```bash
./setup.sh                                   # fetches what is not in the repo (KataCR, datasets) and prepares envs
python scripts/autoplay.py --games 5 --show  # plays 5 ladder matches with a live window
python scripts/match_report.py               # -> runs/report.html: why games were won or lost
```

Other useful commands:

```bash
python scripts/play_smart.py                     # play one battle live (or --dry video.mp4 for decisions only)
python scripts/replay_logs.py --games runs/games # replay logged matches with the current brain, see what changes
python scripts/pack_samples.py                   # gather captures + video moves to push for detector/imitation work
```

**Hardware:** playing needs a decent PC; an NVIDIA GPU makes detection ~40× faster (8.5 ms vs ~350 ms per loop on CPU) and is needed for training.

## Tests

No phone or GPU needed; CI runs them on every pull request.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-test.txt
python -m pytest -q tests
```

## Status and roadmap

1. ✅ Stream, verified taps, elixir and hand reading
2. ✅ Own fast detector (YOLO11s, TensorRT) with tracking, side/name voting and timestamped speeds
3. ✅ Rule brain with real card stats, opponent model, elixir advantage, endgame logic, learned placement
4. ✅ Strategy learning across matches (factorized bandit), match reports, replay of real logs
5. ⏳ Read the in-game clock; calibrate detection colours and retrain on hard examples (needs captures)
6. ⏳ Imitation learning of *what* and *when* to play from pro videos
7. ⏳ Reinforcement learning from our own friendly battles

`HANDOFF.md` (French) holds the detailed history, measurements and the lessons learned along the way.
