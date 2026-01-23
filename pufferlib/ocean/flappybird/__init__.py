"""Flappy Bird environment for PufferLib Ocean."""
from .flappybird import FlappyBird

try:
    import torch
except ImportError:
    pass
else:
    from .policy import Policy
