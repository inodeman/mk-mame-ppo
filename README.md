# MK MAME PPO

PPO reinforcement learning agent to play **Mortal Kombat** (arcade/Genesis) using the **MAME** emulator with parallel environments.

## Goal
Train a bot that learns to play MK in a few hours using 6 simultaneous MAME instances.

## Requirements
- Python 3.10+
- MAME installed and in PATH (or set path in code)
- Mortal Kombat ROM (you must own a legal copy)
- GPU recommended for faster training

```bash
pip install -r requirements.txt
```

## Quick Start
1. Place your MK ROM in `roms/` (or configure path).
2. Find memory addresses for health, round, etc. using MAME debugger or cheats.
3. Adapt `mk_env.py` to talk to your MAME instance (see notes below).
4. Train:

```bash
python train.py --n-envs 6 --timesteps 2000000
```

## Environment Integration Options

### Option A: MAMEToolkit (Linux, recommended if available)
```bash
pip install MAMEToolkit
```
See their docs for Street Fighter examples — adapt for MK.

### Option B: Custom Lua + Socket Bridge (works on Windows/Linux)
- Launch MAME with a Lua script that:
  - Reads RAM (health, positions, state)
  - Sends frame or features over TCP/UDP
  - Receives actions and injects them
- Python side connects as a client/server.

### Option C: Screen capture + input (simplest but slower)
- Use `mss` or `pyautogui` for frames
- Send keys/joystick via `pynput` or MAME Lua

## Recommended Reward Shaping
- +1.0 * damage_dealt (normalized)
- -0.5 * damage_taken
- +50 for round win
- +200 for match win
- -0.001 per step (time penalty)

## Parallel Training
The `train.py` uses Stable-Baselines3 with `SubprocVecEnv` for 6 parallel MAME instances. This is the fastest way to gather experience.

## Next Steps
1. Implement real MAME communication in `mk_env.py`
2. Add curriculum (start vs easy AI)
3. Optional: behavioral cloning warm-start from human demos
4. Monitor with TensorBoard: `tensorboard --logdir logs/`

## Disclaimer
Only use with ROMs you legally own. This is for research/education.
