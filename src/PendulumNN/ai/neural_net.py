from functools import cache
from itertools import pairwise

import pygame
import torch

from torch import nn

from PendulumNN.common import Colors, Context


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
        self._hidden_dims = hidden_dims
        self._layers = NeuralNetwork._build_stack(
            input_dim,
            hidden_dims,
            output_dim,
        )
        self.out: torch.Tensor

        self._optimizer = torch.optim.AdamW(  # type: ignore
            self._layers.parameters(),
            lr=1e-3,
            # weight_decay=0.1,
        )

        self.drawer = _NeuralNetworkDrawer(
            [self._input_dim] + self._hidden_dims + [self._output_dim]
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

    def draw(self, ctx: Context, offset: tuple[int, int]) -> None:
        linears = [layer for layer in self._layers if isinstance(layer, nn.Linear)]
        self.drawer.draw(ctx, offset, linears, self.out)

    def forward(self, x):
        self.out = self._layers(x)
        return self.out

    def update(self, loss: torch.Tensor) -> None:
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()


class _NeuralNetworkDrawer:
    MAX_INTERVAL_Y = 10

    def __init__(self, dims: list[int]) -> None:
        self._dims = dims

    def draw(
        self,
        ctx: Context,
        offset: tuple[int, int],
        linears: list[nn.Linear],
        out: torch.Tensor,
    ) -> None:
        xs = self._xs(ctx.width, offset[0])
        ys = self._ys(ctx.height, offset[1])

        def _line(i: int, j: int, k: int, color: Colors = Colors.BEIGE) -> None:
            _in = (xs[i], ys[i][j])
            _out = (xs[i + 1], ys[i + 1][k])
            pygame.draw.line(ctx.surface, color, _in, _out, width=1)

        for i in range(len(xs) - 1):
            weights = linears[i].weight.detach().cpu().numpy()
            for j in range(len(ys[i])):
                for k in range(len(ys[i + 1])):
                    color = Colors.RED if weights[k, j] <= 0.0 else Colors.GREEN
                    _line(i, j, k, color)

        for i, x in enumerate(xs[:-1]):
            for y in ys[i]:
                pygame.draw.circle(ctx.surface, Colors.BEIGE, (x, y), radius=4)

        for i, y in enumerate(ys[-1]):
            color = Colors.RED if out[i] <= 0 else Colors.GREEN
            pygame.draw.circle(ctx.surface, color, (xs[-1], y), radius=4)

    @cache
    def _xs(self, screen_width: int, offset: int) -> list[int]:
        width = screen_width - offset
        interval_x = width // (len(self._dims) + 1)
        cx = (width - interval_x * (len(self._dims) - 1)) // 2
        return [cx + offset + interval_x * i for i in range(len(self._dims))]

    @cache
    def _ys(self, screen_height: int, offset: int) -> list[list[int]]:
        height = screen_height - offset

        result = []
        for dim in self._dims:
            interval = max(self.MAX_INTERVAL_Y, height // (dim + 1))
            col_offset = (height - (dim - 1) * interval) // 2
            column = [offset + col_offset + i * interval for i in range(dim)]
            result.append(column)
        return result
