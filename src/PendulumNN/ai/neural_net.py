import torch

import numpy as np
from torch import nn

from PendulumNN.simulation import Stats


class NeuralNetwork(nn.Module):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {DEVICE} device")

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self._input_dim = input_dim
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self._input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 3),
        )

    def loss(self, stats: Stats) -> np.float64:
        angle_loss = np.sum(1.0 + stats.cos_angles)
        far_from_center = np.abs(stats.cart_x)
        velocity_penalty = np.sum(np.abs(stats.velocities))
        cart_velocity_penalty = np.abs(stats.cart_velocity)
        return (
            angle_loss
            + 0.1 * far_from_center
            + 0.05 * velocity_penalty
            + 0.05 * cart_velocity_penalty
        )
