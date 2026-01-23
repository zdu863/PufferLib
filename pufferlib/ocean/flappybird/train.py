#!/usr/bin/env python3
"""Training script for Flappy Bird using PPO.

This is a minimal, self-contained PPO implementation designed to train
a small policy on CPU quickly. It saves checkpoints that can be loaded
by autoplay.py for evaluation and replay export.

Usage:
    python train.py                    # Train with defaults
    python train.py --total-steps 500000  # Train for 500k steps
    python train.py --checkpoint model.pt # Custom checkpoint name
"""

import os
import sys
import time
import argparse
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from pufferlib.ocean.flappybird.flappybird import FlappyBird
from pufferlib.ocean.flappybird.policy import Policy


@dataclass
class TrainConfig:
    """Training configuration - tuned via hyperparameter sweep.
    
    These hyperparameters were found via Bayesian optimization (Protein sweep):
    - Higher learning rate (1.3e-3 vs 3e-4): Faster learning
    - Higher gamma (0.995): Better long-term credit assignment
    - Lower gae_lambda (0.91): More bias, less variance
    - More epochs (8): More gradient updates per batch
    
    Results: avg score 6.6, max score 39 at 1M steps
    """
    # Environment  
    num_envs: int = 32
    seed: int = 42
    
    # Training - tuned via sweep
    total_steps: int = 200_000
    num_steps: int = 128
    num_epochs: int = 8     # Tuned (was 4)
    num_minibatches: int = 4
    
    # PPO hyperparameters - tuned via sweep
    learning_rate: float = 1.3e-3   # Tuned (was 3e-4)
    gamma: float = 0.995            # Tuned (was 0.99)
    gae_lambda: float = 0.91        # Tuned (was 0.95)
    clip_coef: float = 0.22         # Tuned (was 0.2)
    ent_coef: float = 0.003         # Tuned (was 0.01)
    vf_coef: float = 0.64           # Tuned (was 0.5)
    max_grad_norm: float = 0.5
    
    # Policy
    hidden_size: int = 64
    
    # Checkpointing
    checkpoint: str = "flappybird_policy.pt"
    save_interval: int = 10
    
    # Logging
    log_interval: int = 1


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Initialize layer weights."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def train(config: TrainConfig):
    """Main training loop."""
    # Set seeds
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    device = torch.device("cpu")
    
    # Create environment
    env = FlappyBird(
        num_envs=config.num_envs,
        seed=config.seed,
        report_interval=10,
    )
    
    # Create policy
    policy = Policy(env, hidden_size=config.hidden_size).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=config.learning_rate, eps=1e-5)
    
    # Calculate batch sizes
    batch_size = config.num_envs * config.num_steps
    minibatch_size = batch_size // config.num_minibatches
    num_updates = config.total_steps // batch_size
    
    print(f"Training Flappy Bird Policy")
    print(f"  Total steps: {config.total_steps:,}")
    print(f"  Num environments: {config.num_envs}")
    print(f"  Steps per rollout: {config.num_steps}")
    print(f"  Batch size: {batch_size}")
    print(f"  Minibatch size: {minibatch_size}")
    print(f"  Num updates: {num_updates}")
    print(f"  Hidden size: {config.hidden_size}")
    print()
    
    # Storage
    obs = torch.zeros((config.num_steps, config.num_envs) + env.single_observation_space.shape).to(device)
    actions = torch.zeros((config.num_steps, config.num_envs)).to(device)
    logprobs = torch.zeros((config.num_steps, config.num_envs)).to(device)
    rewards = torch.zeros((config.num_steps, config.num_envs)).to(device)
    dones = torch.zeros((config.num_steps, config.num_envs)).to(device)
    values = torch.zeros((config.num_steps, config.num_envs)).to(device)
    
    # Initialize
    next_obs, _ = env.reset()
    next_obs = torch.from_numpy(next_obs).to(device)
    next_done = torch.zeros(config.num_envs).to(device)
    
    global_step = 0
    start_time = time.time()
    best_score = -float('inf')
    recent_scores = []
    
    for update in range(1, num_updates + 1):
        # Collect rollout
        for step in range(config.num_steps):
            global_step += config.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            
            with torch.no_grad():
                action_logits, value = policy(next_obs)
                probs = Categorical(logits=action_logits)
                action = probs.sample()
                logprob = probs.log_prob(action)
                
            actions[step] = action
            logprobs[step] = logprob
            values[step] = value.flatten()
            
            # Execute action (C environment expects float32)
            next_obs_np, reward, terminations, truncations, infos = env.step(action.cpu().numpy().astype(np.float32))
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device)
            next_obs = torch.from_numpy(next_obs_np).to(device)
            next_done = torch.from_numpy(next_done).float().to(device)
            
            # Log episode info
            for info in infos:
                if 'score' in info:
                    recent_scores.append(info['score'])
                    
        # Compute GAE
        with torch.no_grad():
            _, next_value = policy(next_obs)
            next_value = next_value.flatten()
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            
            for t in reversed(range(config.num_steps)):
                if t == config.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                    
                delta = rewards[t] + config.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + config.gamma * config.gae_lambda * nextnonterminal * lastgaelam
                
            returns = advantages + values
            
        # Flatten batches
        b_obs = obs.reshape((-1,) + env.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        
        # PPO update
        b_inds = np.arange(batch_size)
        clipfracs = []
        
        for epoch in range(config.num_epochs):
            np.random.shuffle(b_inds)
            
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]
                
                action_logits, newvalue = policy(b_obs[mb_inds])
                probs = Categorical(logits=action_logits)
                newlogprob = probs.log_prob(b_actions[mb_inds])
                entropy = probs.entropy()
                
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                
                with torch.no_grad():
                    clipfracs.append(((ratio - 1.0).abs() > config.clip_coef).float().mean().item())
                    
                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value loss
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                
                # Entropy loss
                entropy_loss = entropy.mean()
                
                # Total loss
                loss = pg_loss - config.ent_coef * entropy_loss + config.vf_coef * v_loss
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
                optimizer.step()
                
        # Logging
        if update % config.log_interval == 0:
            elapsed = time.time() - start_time
            sps = global_step / elapsed
            
            avg_score = np.mean(recent_scores) if recent_scores else 0
            max_score = max(recent_scores) if recent_scores else 0
            
            print(f"Update {update}/{num_updates} | "
                  f"Steps: {global_step:,} | "
                  f"SPS: {sps:.0f} | "
                  f"Avg Score: {avg_score:.2f} | "
                  f"Max Score: {max_score:.0f} | "
                  f"Loss: {loss.item():.4f}")
                  
            if avg_score > best_score and len(recent_scores) > 10:
                best_score = avg_score
                
            recent_scores = []
            
        # Checkpointing
        if update % config.save_interval == 0 or update == num_updates:
            checkpoint_data = {
                'policy_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config.__dict__,
                'global_step': global_step,
                'update': update,
            }
            torch.save(checkpoint_data, config.checkpoint)
            print(f"  Saved checkpoint to {config.checkpoint}")
            
    env.close()
    
    print(f"\nTraining complete!")
    print(f"  Total steps: {global_step:,}")
    print(f"  Total time: {time.time() - start_time:.1f}s")
    print(f"  Best average score: {best_score:.2f}")
    print(f"  Checkpoint saved to: {config.checkpoint}")
    
    return policy


def main():
    parser = argparse.ArgumentParser(description="Train Flappy Bird policy (optimized for speed)")
    parser.add_argument("--num-envs", type=int, default=32, help="Number of parallel environments")
    parser.add_argument("--total-steps", type=int, default=200_000, help="Total training steps")
    parser.add_argument("--num-steps", type=int, default=128, help="Steps per rollout")
    parser.add_argument("--num-epochs", type=int, default=4, help="PPO epochs per update")
    parser.add_argument("--hidden-size", type=int, default=64, help="Hidden layer size")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--checkpoint", type=str, default="flappybird_policy.pt", help="Checkpoint filename")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    config = TrainConfig(
        num_envs=args.num_envs,
        total_steps=args.total_steps,
        num_steps=args.num_steps,
        num_epochs=args.num_epochs,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        checkpoint=args.checkpoint,
        seed=args.seed,
    )
    
    train(config)


if __name__ == "__main__":
    main()
