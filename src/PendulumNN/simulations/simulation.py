from abc import ABC, abstractmethod

from PendulumNN.common import Context


class Simulation(ABC):
    @abstractmethod
    def update(self) -> None: ...

    @abstractmethod
    def draw(self, ctx: Context) -> None: ...
