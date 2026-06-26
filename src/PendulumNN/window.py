from __future__ import annotations

import pygame

from PendulumNN.models import Colors
from PendulumNN.pendulum import PendulumSimulation, UpdateStrategy, Stats


class Window:
    def __init__(self, width: int, height: int) -> None:
        self._width = width
        self._height = height

        self._running: bool = True

        self._screen: pygame.Surface | None = None

        self._pendulum = PendulumSimulation(
            self._width // 2,
            self._height // 2,
            nodes=3,
            pendulum_length=50,
            damping=0.5,
            update_strategy=UpdateStrategy.EULER,
        )

    @property
    def screen(self) -> pygame.Surface:
        if self._screen is None:
            raise ValueError("self._screen should be assigned")
        return self._screen

    @screen.setter
    def screen(self, value: pygame.Surface) -> None:
        self._screen = value

    def __enter__(self) -> Window:
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((self._width, self._height))
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        _ = exc_type, exc_value, traceback
        pygame.quit()
        return False

    def run(self) -> None:
        while self._running:
            self._check_event()
            self._update()
            self._draw()
            stats: Stats = self._pendulum.stats

    def _check_event(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self._pendulum.accelerate_left()
        if keys[pygame.K_RIGHT]:
            self._pendulum.accelerate_right()

    def _update(self) -> None:
        self._pendulum.update()

    def _draw(self) -> None:
        self.screen.fill(Colors.GREY)
        self._pendulum.draw(self.screen)
        pygame.display.update()
