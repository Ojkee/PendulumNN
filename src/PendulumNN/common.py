from __future__ import annotations
from enum import Enum


class Colors(tuple[int, int, int], Enum):
    BEIGE = (255, 248, 231)
    GREY = (51, 51, 51)


class Action(Enum):
    LEFT = 0
    RIGHT = 1
    STAY = 2
