from __future__ import annotations
from enum import Enum
from dataclasses import dataclass

import pygame


class Colors(tuple[int, int, int], Enum):
    BEIGE = (255, 248, 231)
    GREY = (51, 51, 51)


class Action(Enum):
    LEFT = 0
    RIGHT = 1
    STAY = 2


@dataclass
class Context:
    width: int
    height: int
    _surface: pygame.Surface | None

    @property
    def surface(self) -> pygame.Surface:
        if self._surface is None:
            raise ValueError("surface not set")
        return self._surface

    @surface.setter
    def surface(self, value: pygame.Surface) -> None:
        self._surface = value
