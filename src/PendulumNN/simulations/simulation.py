from abc import ABC, abstractmethod

import numpy as np

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

    @abstractmethod
    def fitness(self) -> np.float64: ...
