#!/usr/bin/env python3
"""Hyperparameter sweep for Flappy Bird using PufferLib's built-in tuner.

Uses Gaussian Process-based Bayesian optimization (Protein method) to find
optimal hyperparameters for PPO training.

Usage:
    python sweep.py --num-trials 20 --total-steps 500000
"""

import os
import sys
import time
import random
import argparse
from dataclasses import dataclass
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import pufferlib.sweep as sweep
from pufferlib.ocean.flappybird.flappybird import FlappyBird
from pufferlib.ocean.flappybird.policy import Policy


# Sweep configuration
SWEEP_CONFIG = {
    'method': 'Protein',
    'metric': 'score',
    'metric_distribution': 'linear',
    'goal': 'maximize',
    'downsample': 1,
    'use_gpu': False,
    'prune_pareto': True,
    'early_stop_quantile': 0.3,
    'max_suggestion_cost': 600,  # Max seconds per trial
    
    # Hyperparameters to tune
    'train': {
        'learning_rate': {
            'distribution': 'log_normal',
            'min': 1e-5,
            'max': 1e-2,
            'scale': 0.5,
        },
        'gamma': {
            'distribution': 'logit_normal',
            'min': 0.9,
            'max': 0.999,
            'scale': 0.5,
        },
        'gae_lambda': {
            'distribution': 'logit_normal',
            'min': 0.8,
            'max': 0.99,
            'scale': 0.5,
        },
        'clip_coef': {
            'distribution': 'uniform',
            'min': 0.1,
            'max': 0.4,
            'scale': 0.5,
        },
        'ent_coef': {
            'distribution': 'log_normal',
            'min': 0.001,
            'max': 0.1,
            'scale': 0.5,
        },
        'vf_coef': {
            'distribution': 'uniform',
            'min': 0.1,
            'max': 1.0,
            'scale': 0.5,
        },
        'num_epochs': {
            'distribution': 'int_uniform',
            'min': 1,
            'max': 8,
            'scale': 0.5,
        },
        'num_minibatches': {
            'distribution': 'int_uniform',
            'min': 1,
            'max': 8,
            'scale': 0.5,
        },
    }
}


def train_with_params(params, total_steps=500_000, num_envs=32, seed=None):
    """Train with given hyperparameters and return score."""
    if seed is None:
        seed = int(time.time()) % 10000
        
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    train_params = params.get('train', params)
    
    # Extract hyperparameters
    learning_rate = train_params.get('learning_rate', 3e-4)
    gamma = train_params.get('gamma', 0.99)
    gae_lambda = train_params.get('gae_lambda', 0.95)
    clip_coef = train_params.get('clip_coef', 0.2)
    ent_coef = train_params.get('ent_coef', 0.01)
    vf_coef = train_params.get('vf_coef', 0.5)
    num_epochs = int(train_params.get('num_epochs', 4))
    num_minibatches = int(train_params.get('num_minibatches', 4))
    
    num_steps = 128
    batch_size = num_envs * num_steps
    minibatch_size = batch_size // num_minibatches
    
    device = torch.device("cpu")
    
    # Create environment and policy
    env = FlappyBird(num_envs=num_envs, seed=seed, report_interval=10)
    policy = Policy(env, hidden_size=64).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate, eps=1e-5)
    
    # Storage
    obs = torch.zeros((num_steps, num_envs) + env.single_observation_space.shape).to(device)
    actions = torch.zeros((num_steps, num_envs)).to(device)
    logprobs = torch.zeros((num_steps, num_envs)).to(device)
    rewards = torch.zeros((num_steps, num_envs)).to(device)
    dones = torch.zeros((num_steps, num_envs)).to(device)
    values = torch.zeros((num_steps, num_envs)).to(device)
    
    # Initialize
    next_obs, _ = env.reset()
    next_obs = torch.from_numpy(next_obs).to(device)
    next_done = torch.zeros(num_envs).to(device)
    
    global_step = 0
    all_scores = []
    start_time = time.time()
    
    num_updates = total_steps // batch_size
    
    for update in range(1, num_updates + 1):
        # Collect rollout
        for step in range(num_steps):
            global_step += num_envs
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
            
            next_obs_np, reward, terminations, truncations, infos = env.step(
                action.cpu().numpy().astype(np.float32))
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device)
            next_obs = torch.from_numpy(next_obs_np).to(device)
            next_done = torch.from_numpy(next_done).float().to(device)
            
            for info in infos:
                if 'score' in info:
                    all_scores.append(info['score'])
                    
        # Compute GAE
        with torch.no_grad():
            _, next_value = policy(next_obs)
            next_value = next_value.flatten()
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                    
                delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
                
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
        
        for epoch in range(num_epochs):
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
                
                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value loss
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                
                # Entropy loss
                entropy_loss = entropy.mean()
                
                # Total loss
                loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()
                
    env.close()
    
    # Return average score from last 25% of training
    elapsed = time.time() - start_time
    if all_scores:
        recent_scores = all_scores[-(len(all_scores) // 4):]
        avg_score = np.mean(recent_scores) if recent_scores else 0
        max_score = max(all_scores)
    else:
        avg_score, max_score = 0, 0
        
    return avg_score, max_score, elapsed, all_scores


def run_sweep(num_trials=20, total_steps=500_000, num_envs=32):
    """Run hyperparameter sweep."""
    print("="*60)
    print("Flappy Bird Hyperparameter Sweep")
    print("="*60)
    print(f"Trials: {num_trials}")
    print(f"Steps per trial: {total_steps:,}")
    print(f"Environments: {num_envs}")
    print()
    
    # Initialize sweep
    sweeper = sweep.Protein(
        SWEEP_CONFIG,
        max_suggestion_cost=600,
        use_gpu=False,
    )
    
    best_score = -float('inf')
    best_params = None
    results = []
    
    for trial in range(num_trials):
        print(f"\n{'='*60}")
        print(f"Trial {trial + 1}/{num_trials}")
        print("="*60)
        
        # Get suggested hyperparameters
        params, info = sweeper.suggest(fill=deepcopy(SWEEP_CONFIG))
        
        train_params = params.get('train', params)
        print("Hyperparameters:")
        for key, val in train_params.items():
            if isinstance(val, dict):
                continue
            if isinstance(val, float):
                print(f"  {key}: {val:.6f}")
            else:
                print(f"  {key}: {val}")
        
        # Train with these parameters
        try:
            avg_score, max_score, elapsed, scores = train_with_params(
                params, total_steps=total_steps, num_envs=num_envs
            )
            
            print(f"\nResults:")
            print(f"  Avg Score: {avg_score:.2f}")
            print(f"  Max Score: {max_score:.0f}")
            print(f"  Time: {elapsed:.1f}s")
            
            # Report to sweeper
            sweeper.observe(params, avg_score, elapsed)
            
            results.append({
                'trial': trial,
                'params': train_params,
                'avg_score': avg_score,
                'max_score': max_score,
                'time': elapsed,
            })
            
            if avg_score > best_score:
                best_score = avg_score
                best_params = train_params.copy()
                print(f"  *** New best! ***")
                
        except Exception as e:
            print(f"Trial failed: {e}")
            sweeper.observe(params, 0, 0, is_failure=True)
            
    print("\n" + "="*60)
    print("SWEEP COMPLETE")
    print("="*60)
    print(f"\nBest Score: {best_score:.2f}")
    print("\nBest Hyperparameters:")
    if best_params:
        for key, val in best_params.items():
            if isinstance(val, dict):
                continue
            if isinstance(val, float):
                print(f"  {key}: {val:.6f}")
            else:
                print(f"  {key}: {val}")
                
    return best_params, results


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter sweep for Flappy Bird")
    parser.add_argument("--num-trials", type=int, default=20, help="Number of trials")
    parser.add_argument("--total-steps", type=int, default=500_000, help="Steps per trial")
    parser.add_argument("--num-envs", type=int, default=32, help="Parallel environments")
    
    args = parser.parse_args()
    
    best_params, results = run_sweep(
        num_trials=args.num_trials,
        total_steps=args.total_steps,
        num_envs=args.num_envs,
    )
    
    # Save results
    if results:
        import json
        with open('sweep_results.json', 'w') as f:
            json.dump({
                'best_params': {k: v for k, v in best_params.items() if not isinstance(v, dict)},
                'results': [{
                    'trial': r['trial'],
                    'avg_score': r['avg_score'],
                    'max_score': r['max_score'],
                    'time': r['time'],
                    'params': {k: v for k, v in r['params'].items() if not isinstance(v, dict)},
                } for r in results],
            }, f, indent=2)
        print("\nResults saved to sweep_results.json")


if __name__ == "__main__":
    main()
