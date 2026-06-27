from abc import ABC, abstractmethod

from PendulumNN.common import Context


class Screen(ABC):
    @abstractmethod
    def handle_event(self) -> None: ...

    @abstractmethod
    def update(self) -> None: ...

    @abstractmethod
    def draw(self, ctx: Context) -> None: ...
