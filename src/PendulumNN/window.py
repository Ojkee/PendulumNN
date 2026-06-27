from __future__ import annotations

import pygame

from PendulumNN.common import Colors, Context
from PendulumNN.ai.neural_net import NeuralNetwork
from PendulumNN.simulation import PendulumSimulation, UpdateStrategy


class Window:
    def __init__(self, width: int, height: int) -> None:
        self.ctx = Context(width, height, _surface=None)
        self._running: bool = True
        self._pendulum = PendulumSimulation(
            nodes=1,
            pendulum_length=50,
            damping=0.25,
            update_strategy=UpdateStrategy.EULER,
        )
        self.model = NeuralNetwork(
            input_dim=len(self._pendulum.stats.as_flat()),
        ).to(NeuralNetwork.DEVICE)

    def __enter__(self) -> Window:
        pygame.init()
        pygame.font.init()
        self.ctx.surface = pygame.display.set_mode((self.ctx.width, self.ctx.height))
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
        self.ctx.surface.fill(Colors.GREY)
        self._pendulum.draw(self.ctx)
