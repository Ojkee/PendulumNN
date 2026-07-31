from itertools import pairwise

import torch

from torch import nn

from PendulumNN.common import Context


class NeuralNetwork(nn.Module):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {DEVICE} device")

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
    ) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._layers = NeuralNetwork._build_stack(
            input_dim,
            hidden_dims,
            output_dim,
        )
        self.out: torch.Tensor

        self._optimizer = torch.optim.AdamW(  # type: ignore
            self._layers.parameters(), lr=5e-2, weight_decay=0.1
        )

    @staticmethod
    def _build_stack(
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
    ) -> nn.Sequential:
        if len(hidden_dims) == 0:
            return nn.Sequential(nn.Linear(input_dim, output_dim))
        stack = [
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
        ]
        for inn, out in pairwise(hidden_dims):
            stack.extend([nn.Linear(inn, out), nn.ReLU()])
        stack.append(nn.Linear(hidden_dims[-1], output_dim))
        return nn.Sequential(*stack)

    def draw(self, ctx: Context) -> None:
        _ = ctx
        pass

    def forward(self, x):
        self.out = self._layers(x)
        return self.out

    def update(self, loss: torch.Tensor) -> None:
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
