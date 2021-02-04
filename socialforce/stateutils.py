"""Utility functions to process state."""

import torch


def desired_directions(state):
    """Given the current state and destination, compute desired direction."""
    destination_vectors = state[:, 6:8] - state[:, 0:2]
    norm_factors = torch.norm(destination_vectors, dim=-1)
    norm_factors[norm_factors == 0.0] = 1.0
    return destination_vectors / norm_factors.unsqueeze(-1)


def speeds(state):
    """Return the speeds corresponding to a given state."""
    return torch.norm(state[:, 2:4], dim=-1)
