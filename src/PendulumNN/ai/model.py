from typing import Protocol

import torch

from PendulumNN.common import Context


class AIModel(Protocol):
    def __call__(self, x) -> torch.Tensor: ...
    def draw(self, ctx: Context, offset: tuple[int, int]) -> None: ...
    def update(self, loss: torch.Tensor) -> None: ...
