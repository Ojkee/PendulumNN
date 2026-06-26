from enum import Enum
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from PendulumNN.pendulum import Stats


class Prediction(Enum):
    LEFT = 0
    RIGHT = 1
    STAY = 2


class NeuralNetwork(nn.Module):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {DEVICE} device")

    def __init__(self, input_dim: int, hidden_layers: int) -> None:
        super().__init__()
        self._hidden_layers = hidden_layers
        self._input_dim = input_dim
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self._input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_relu_stack(x)

    def act(self, stats_flat: np.ndarray) -> tuple[Prediction, torch.Tensor]:
        x = torch.as_tensor(stats_flat, dtype=torch.float32, device=self.DEVICE)
        logits = self.forward(x)
        dist = Categorical(logits=logits)
        action_idx = dist.sample()
        log_prob = dist.log_prob(action_idx)
        return Prediction(int(action_idx.item())), log_prob

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
