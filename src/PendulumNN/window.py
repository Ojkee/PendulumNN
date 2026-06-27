from __future__ import annotations

import pygame

from PendulumNN.common import Colors
from PendulumNN.ai.neural_net import NeuralNetwork
from PendulumNN.simulation import PendulumSimulation, UpdateStrategy


class Window:
    EPISODE_LENGTH = 1000

    def __init__(self, width: int, height: int) -> None:
        self._width = width
        self._height = height
        self._running: bool = True
        self._surface: pygame.Surface | None = None
        self._pendulum = PendulumSimulation(
            self._width // 2,
            self._height // 2,
            nodes=1,
            pendulum_length=50,
            damping=0.25,
            update_strategy=UpdateStrategy.EULER,
        )
        self.model = NeuralNetwork(
            input_dim=len(self._pendulum.stats.as_flat()),
        ).to(NeuralNetwork.DEVICE)

    @property
    def surface(self) -> pygame.Surface:
        if self._surface is None:
            raise ValueError("self._screen should be assigned")
        return self._surface

    @surface.setter
    def surface(self, value: pygame.Surface) -> None:
        self._surface = value

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

    def _check_event(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False

    def _update(self) -> None:
        self._pendulum.update()
        pygame.display.update()

    def _draw(self) -> None:
        self.screen.fill(Colors.GREY)
        self._draw_axis()
        self._pendulum.draw(self.screen)

    def _draw_axis(self) -> None:
        for h in range(100, self._height // 2 + 100, 10):
            pygame.draw.circle(
                self.screen,
                Colors.BEIGE,
                (self._width // 2, h),
                radius=1,
                width=1,
            )
        pygame.draw.line(
            self.screen,
            Colors.BEIGE,
            (0, self._height // 2),
            (self._width, self._height // 2),
            width=1,
        )
