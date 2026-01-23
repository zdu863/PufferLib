# Flappy Bird Environment

A minimal, headless, deterministic Flappy Bird implementation for PufferLib Ocean.

**High-performance C implementation with Python bindings.**

## Features

- **High Performance**: C implementation achieves 80M+ SPS (Steps Per Second)
- **Headless**: No graphics dependencies, runs on CPU
- **Deterministic**: Reproducible with seed
- **Vectorized**: Efficient parallel environment execution
- **Simple Physics**: Gravity + flap velocity
- **Vector Observations**: 8D state (no pixels)
- **Discrete Actions**: 2 actions (flap / no-op)

## Observation Space

8-dimensional vector (Box):
| Index | Description |
|-------|-------------|
| 0 | Bird Y position (normalized) |
| 1 | Bird velocity (normalized) |
| 2 | Distance to next pipe (normalized) |
| 3 | Next pipe gap top Y (normalized) |
| 4 | Next pipe gap bottom Y (normalized) |
| 5 | Distance to second pipe (normalized) |
| 6 | Second pipe gap top Y (normalized) |
| 7 | Second pipe gap bottom Y (normalized) |

## Action Space

Discrete(2):
- 0: No-op (gravity applies)
- 1: Flap (upward velocity)

## Reward

- +0.1: Alive bonus (per step)
- +1.0: Passing through a pipe
- -1.0: Death (collision or out of bounds)

## Usage

### Building (if not already built)

```bash
# From PufferLib root directory
python setup.py build_flappybird --inplace
```

### Training

```bash
cd pufferlib/ocean/flappybird
python train.py --total-steps 2000000 --checkpoint model.pt
```

Options:
- `--num-envs`: Number of parallel environments (default: 32)
- `--total-steps`: Total training steps (default: 200000)
- `--hidden-size`: Policy hidden layer size (default: 64)
- `--learning-rate`: Learning rate (default: 3e-4)
- `--checkpoint`: Output checkpoint file (default: flappybird_policy.pt)
- `--seed`: Random seed (default: 42)

### Evaluation & Replay Export

```bash
python autoplay.py --checkpoint model.pt --episodes 20 --output replay.jsonl
```

Options:
- `--checkpoint`: Path to trained policy checkpoint
- `--episodes`: Number of evaluation episodes (default: 5)
- `--output`: Output replay file in JSONL format (default: replay.jsonl)
- `--seed`: Random seed for reproducibility (default: 12345)
- `--deterministic`: Use argmax instead of sampling

### Web Replay Viewer

The viewer auto-loads `replay.jsonl` from the same directory. You can also:
- Open `viewer.html` in a browser and manually upload a replay file
- Serve the directory with a simple HTTP server for auto-loading

```bash
# Serve the viewer with Python (for auto-loading)
python -m http.server 8000
# Then open http://localhost:8000/viewer.html
```

Keyboard controls:
- Space/K: Play/Pause
- Arrow Left/J: Step back
- Arrow Right/L: Step forward
- R: Restart episode

## Architecture

```
flappybird/
├── __init__.py          # Module exports
├── flappybird.h         # C header (environment logic)
├── binding.c            # Python C extension bindings
├── flappybird.py        # Python wrapper (PufferEnv)
├── policy.py            # PyTorch policy network
├── train.py             # PPO training script
├── autoplay.py          # Evaluation & replay export
├── viewer.html          # Web-based replay viewer
└── README.md            # This file
```

## Environment Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| SCREEN_HEIGHT | 512 | Game screen height |
| SCREEN_WIDTH | 288 | Game screen width |
| GRAVITY | 0.5 | Gravity acceleration |
| FLAP_VELOCITY | -8.0 | Upward velocity on flap |
| PIPE_GAP | 100 | Vertical gap between pipes |
| PIPE_VELOCITY | -4.0 | Horizontal pipe speed |
| MAX_STEPS | 2000 | Episode step limit |

## Performance

- **Environment**: 80,000,000+ SPS (C implementation, 4096 parallel envs)
- **Training**: ~42,000 SPS with PPO on CPU (64 envs)

## Example Results

After 2M training steps (~47 seconds):
- Average Score: ~5 pipes per episode
- Max Score: 14+ pipes
- Best training average: ~5.0 pipes
