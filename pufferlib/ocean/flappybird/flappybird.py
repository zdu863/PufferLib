"""Flappy Bird environment for PufferLib Ocean.

A minimal, headless, deterministic Flappy Bird implementation with:
- Discrete actions: 0 = no-op, 1 = flap
- Vector observations (no pixels)
- Simple physics (gravity/flap)
- Pipes, collisions, score tracking
- Learnable reward: alive bonus + pipe pass bonus - death penalty

Uses C implementation via bindings for high performance.
"""

import numpy as np
import gymnasium
import pufferlib
from pufferlib.ocean.flappybird import binding


class FlappyBird(pufferlib.PufferEnv):
    """Vectorized Flappy Bird environment following PufferLib Ocean standard.
    
    Observation (8 features):
        0: bird_y (normalized to [-1, 1])
        1: bird_velocity (normalized)
        2: dist_to_next_pipe (normalized)
        3: next_pipe_top_y (normalized)
        4: next_pipe_bottom_y (normalized)  
        5: dist_to_second_pipe (normalized)
        6: second_pipe_top_y (normalized)
        7: second_pipe_bottom_y (normalized)
    
    Actions:
        0: no-op (do nothing, gravity applies)
        1: flap (apply upward velocity)
    
    Rewards:
        +0.1: alive bonus each step
        +1.0: passing through a pipe
        -1.0: death (collision or out of bounds)
    """
    
    # Game constants (match C header)
    SCREEN_HEIGHT = 512.0
    SCREEN_WIDTH = 288.0
    BIRD_X = 50.0
    BIRD_RADIUS = 12.0
    PIPE_WIDTH = 52.0
    PIPE_GAP = 100.0
    MAX_STEPS = 2000
    NUM_PIPES = 2
    
    def __init__(
        self,
        num_envs: int = 1,
        render_mode: str = None,
        report_interval: int = 1,
        buf=None,
        seed: int = 0,
    ):
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.report_interval = report_interval
        self.tick = 0
        
        # Observation: 8 features (bird state + 2 pipes info)
        self.num_obs = 8
        self.single_observation_space = gymnasium.spaces.Box(
            low=-2.0, high=2.0, shape=(self.num_obs,), dtype=np.float32
        )
        
        # Action: 0 = no-op, 1 = flap
        self.single_action_space = gymnasium.spaces.Discrete(2)
        
        super().__init__(buf)
        
        # Actions buffer (float32 for C binding)
        self.actions = np.zeros(num_envs, dtype=np.float32)
        
        # Initialize C environments
        self.c_envs = binding.vec_init(
            self.observations,
            self.actions,
            self.rewards,
            self.terminals,
            self.truncations,
            num_envs,
            seed,
        )
        
    def reset(self, seed=None):
        self.tick = 0
        if seed is None:
            binding.vec_reset(self.c_envs, 0)
        else:
            binding.vec_reset(self.c_envs, seed)
        return self.observations, []
        
    def step(self, actions):
        # Copy actions to buffer
        self.actions[:] = actions
        
        self.tick += 1
        binding.vec_step(self.c_envs)
        
        info = []
        if self.tick % self.report_interval == 0:
            log = binding.vec_log(self.c_envs)
            if log:  # Only append if there's data
                info.append(log)
        
        return (
            self.observations,
            self.rewards,
            self.terminals,
            self.truncations,
            info
        )
        
    def render(self):
        binding.vec_render(self.c_envs, 0)
        
    def close(self):
        binding.vec_close(self.c_envs)
        
    def get_state(self, env_idx=0):
        """Get the current state for replay recording.
        
        Reconstructs exact game state from observations.
        The C observations are:
            obs[0] = (bird_y - SCREEN_HEIGHT/2) / (SCREEN_HEIGHT/2)
            obs[1] = bird_velocity / MAX_VELOCITY
            obs[2] = (pipe_x - BIRD_X) / SCREEN_WIDTH
            obs[3] = (gap_y - PIPE_GAP/2 - SCREEN_HEIGHT/2) / (SCREEN_HEIGHT/2)  # top of gap
            obs[4] = (gap_y + PIPE_GAP/2 - SCREEN_HEIGHT/2) / (SCREEN_HEIGHT/2)  # bottom of gap
        """
        obs = self.observations[env_idx]
        
        # Denormalize bird state (exact inverse of C normalization)
        bird_y = obs[0] * (self.SCREEN_HEIGHT / 2) + self.SCREEN_HEIGHT / 2
        bird_velocity = obs[1] * 10.0  # MAX_VELOCITY = 10.0
        
        # Denormalize pipe info
        pipes = []
        for i in range(2):
            # pipe_x from normalized distance
            pipe_x = obs[2 + i * 3] * self.SCREEN_WIDTH + self.BIRD_X
            
            # gap edges from normalized values
            gap_top = obs[3 + i * 3] * (self.SCREEN_HEIGHT / 2) + self.SCREEN_HEIGHT / 2
            gap_bottom = obs[4 + i * 3] * (self.SCREEN_HEIGHT / 2) + self.SCREEN_HEIGHT / 2
            
            # gap center is midpoint of gap_top and gap_bottom
            # gap_top = gap_y - PIPE_GAP/2, gap_bottom = gap_y + PIPE_GAP/2
            # so gap_y = (gap_top + gap_bottom) / 2
            gap_center = (gap_top + gap_bottom) / 2
            
            pipes.append([float(pipe_x), float(gap_center), 0.0])
        
        return {
            'bird_y': float(bird_y),
            'bird_velocity': float(bird_velocity),
            'pipes': pipes,
            'score': 0,  # Score is tracked separately in Python
            'step': self.tick,
        }


def test_performance(timeout=10, num_envs=4096):
    """Benchmark environment performance."""
    env = FlappyBird(num_envs=num_envs)
    env.reset()
    tick = 0
    
    # Pre-generate random actions
    atn_cache = 8192
    actions = np.random.randint(0, 2, (atn_cache, num_envs)).astype(np.float32)
    
    import time
    start = time.time()
    while time.time() - start < timeout:
        atn = actions[tick % atn_cache]
        env.step(atn)
        tick += 1
        
    sps = num_envs * tick / (time.time() - start)
    print(f'SPS: {sps:,.0f}')
    return sps


if __name__ == '__main__':
    # Quick test
    print("Testing FlappyBird environment...")
    env = FlappyBird(num_envs=2, seed=42)
    obs, _ = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Sample observation: {obs[0]}")
    
    # Run a few steps
    for i in range(100):
        actions = np.random.randint(0, 2, 2)
        obs, rewards, terminals, truncations, info = env.step(actions)
        if info:
            print(f"Step {i}: {info}")
            
    print("\nRunning performance test...")
    test_performance(timeout=5, num_envs=1024)
