from abc import ABC, abstractmethod

import torch

from PendulumNN.common import Context


class Simulation(ABC):
    @abstractmethod
    def update(self) -> None: ...

    @abstractmethod
    def draw(self, ctx: Context) -> None: ...

    @property
    @abstractmethod
    def input_dim(self) -> int: ...

    @property
    @abstractmethod
    def output_dim(self) -> int: ...

    @property
    @abstractmethod
    def input_state_vector(self) -> torch.Tensor: ...

    @abstractmethod
    def handle_output_vector(self, y: torch.Tensor) -> torch.Tensor:
        r"""
        Simulation handles output and returns loss
        """

    @abstractmethod
    def reset(self) -> None: ...
