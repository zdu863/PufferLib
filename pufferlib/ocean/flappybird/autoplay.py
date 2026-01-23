#!/usr/bin/env python3
"""Autoplay script for Flappy Bird.

Loads a trained policy, runs evaluation episodes, and exports replay data
to a JSONL file for web visualization.

Usage:
    python autoplay.py                          # Use default checkpoint
    python autoplay.py --checkpoint model.pt   # Custom checkpoint
    python autoplay.py --episodes 10           # Run 10 episodes
    python autoplay.py --output replay.jsonl   # Custom output file
"""

import os
import sys
import json
import argparse
from dataclasses import dataclass

import numpy as np
import torch
from torch.distributions.categorical import Categorical

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from pufferlib.ocean.flappybird.flappybird import FlappyBird
from pufferlib.ocean.flappybird.policy import Policy


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    checkpoint: str = "flappybird_policy.pt"
    episodes: int = 5
    output: str = "replay.jsonl"
    seed: int = 12345
    deterministic: bool = False  # Use argmax instead of sampling


def load_policy(checkpoint_path: str, env):
    """Load policy from checkpoint."""
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please train a policy first using: python train.py")
        sys.exit(1)
        
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Get hidden size from checkpoint config
    config = checkpoint.get('config', {})
    hidden_size = config.get('hidden_size', 64)
    
    policy = Policy(env, hidden_size=hidden_size)
    policy.load_state_dict(checkpoint['policy_state_dict'])
    policy.eval()
    
    print(f"Loaded policy from {checkpoint_path}")
    print(f"  Training steps: {checkpoint.get('global_step', 'unknown'):,}")
    print(f"  Hidden size: {hidden_size}")
    
    return policy


def run_episode(env, policy, deterministic=False):
    """Run a single episode and record frames for replay.
    
    Returns:
        frames: List of frame data for replay
        score: Final score (tracked via rewards)
        steps: Number of steps
    """
    frames = []
    
    obs, _ = env.reset()
    obs_tensor = torch.from_numpy(obs)
    
    done = False
    step = 0
    score = 0  # Track score via rewards
    
    while not done:
        # Get action from policy
        with torch.no_grad():
            action_logits, value = policy(obs_tensor)
            
            if deterministic:
                action = action_logits.argmax(dim=-1)
            else:
                probs = Categorical(logits=action_logits)
                action = probs.sample()
                
        action_np = action.numpy().astype(np.float32)
        
        # Record frame before step (using observation-based state)
        state = env.get_state(0)
        frame = {
            'step': step,
            'bird_y': state['bird_y'],
            'bird_velocity': state['bird_velocity'],
            'pipes': state['pipes'],
            'action': int(action_np[0]),
            'score': score,
        }
        frames.append(frame)
        
        # Store last state for final frame
        last_bird_y = state['bird_y']
        last_bird_velocity = state['bird_velocity']
        last_pipes = state['pipes']
        
        # Step environment
        obs, rewards, terminals, truncations, infos = env.step(action_np)
        obs_tensor = torch.from_numpy(obs)
        
        done = terminals[0] or truncations[0]
        step += 1
        
        # Track score via rewards (1.0 for pipe pass)
        if rewards[0] > 0.5:
            score += 1
        
    # Record final frame
    final_frame = {
        'step': step,
        'bird_y': last_bird_y,
        'bird_velocity': last_bird_velocity,
        'pipes': last_pipes,
        'action': -1,  # No action (episode ended)
        'score': score,
        'terminal': True,
    }
    frames.append(final_frame)
    
    return frames, score, step


def export_replay(episodes_data: list, output_path: str):
    """Export replay data to JSONL file.
    
    Format: One JSON object per line, with metadata line first.
    """
    with open(output_path, 'w') as f:
        # Write metadata
        metadata = {
            'type': 'metadata',
            'game': 'flappybird',
            'version': '1.0',
            'num_episodes': len(episodes_data),
            'constants': {
                'screen_height': FlappyBird.SCREEN_HEIGHT,
                'screen_width': FlappyBird.SCREEN_WIDTH,
                'bird_x': FlappyBird.BIRD_X,
                'bird_radius': FlappyBird.BIRD_RADIUS,
                'pipe_width': FlappyBird.PIPE_WIDTH,
                'pipe_gap': FlappyBird.PIPE_GAP,
            }
        }
        f.write(json.dumps(metadata) + '\n')
        
        # Write episodes
        for ep_idx, episode in enumerate(episodes_data):
            episode_data = {
                'type': 'episode',
                'episode_id': ep_idx,
                'score': episode['score'],
                'steps': episode['steps'],
                'frames': episode['frames'],
            }
            f.write(json.dumps(episode_data) + '\n')
            
    print(f"Exported replay to {output_path}")


def evaluate(config: EvalConfig):
    """Run evaluation and export replays."""
    # Set seed for reproducibility
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    # Create environment (single env for replay recording)
    env = FlappyBird(num_envs=1, seed=config.seed)
    
    # Load policy
    policy = load_policy(config.checkpoint, env)
    
    print(f"\nRunning {config.episodes} evaluation episodes...")
    print(f"Deterministic mode: {config.deterministic}")
    print()
    
    episodes_data = []
    scores = []
    steps_list = []
    
    for ep_idx in range(config.episodes):
        frames, score, steps = run_episode(env, policy, config.deterministic)
        
        episodes_data.append({
            'frames': frames,
            'score': score,
            'steps': steps,
        })
        
        scores.append(score)
        steps_list.append(steps)
        
        print(f"Episode {ep_idx + 1}: Score={score}, Steps={steps}")
        
    env.close()
    
    # Print summary
    print(f"\n{'='*50}")
    print("Evaluation Summary")
    print(f"{'='*50}")
    print(f"Episodes: {config.episodes}")
    print(f"Average Score: {np.mean(scores):.2f} (+/- {np.std(scores):.2f})")
    print(f"Max Score: {max(scores)}")
    print(f"Min Score: {min(scores)}")
    print(f"Average Steps: {np.mean(steps_list):.1f}")
    print(f"{'='*50}")
    
    # Export replay
    export_replay(episodes_data, config.output)
    
    return scores


def main():
    parser = argparse.ArgumentParser(description="Evaluate Flappy Bird policy and export replay")
    parser.add_argument("--checkpoint", type=str, default="flappybird_policy.pt", 
                        help="Path to policy checkpoint")
    parser.add_argument("--episodes", type=int, default=5, 
                        help="Number of evaluation episodes")
    parser.add_argument("--output", type=str, default="replay.jsonl", 
                        help="Output replay file (JSONL format)")
    parser.add_argument("--seed", type=int, default=12345, 
                        help="Random seed for reproducibility")
    parser.add_argument("--deterministic", action="store_true", 
                        help="Use argmax instead of sampling (deterministic mode)")
    
    args = parser.parse_args()
    
    config = EvalConfig(
        checkpoint=args.checkpoint,
        episodes=args.episodes,
        output=args.output,
        seed=args.seed,
        deterministic=args.deterministic,
    )
    
    evaluate(config)


if __name__ == "__main__":
    main()
