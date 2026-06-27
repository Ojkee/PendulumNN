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

    @property
    @abstractmethod
    def input_state_vector(self) -> np.ndarray: ...

    @abstractmethod
    def handle_output_vector(self, y: np.ndarray) -> None: ...

    @abstractmethod
    def fitness(self, ctx: Context) -> np.float64: ...

    @abstractmethod
    def reset(self) -> None: ...
