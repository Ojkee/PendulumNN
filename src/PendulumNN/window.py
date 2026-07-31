from __future__ import annotations

import pygame

from PendulumNN.ai.neural_net import NeuralNetwork
from PendulumNN.common import Colors, Context
from PendulumNN.screens import Screen, SimulationScreen
from PendulumNN.simulations import PendulumSimulation


class Window:
    def __init__(self, width: int, height: int) -> None:
        self.ctx = Context(width, height)
        self._running: bool = True
        _simulation = PendulumSimulation(nodes=1, pendulum_length=50, damping=0.25)
        self._current_screen: Screen = SimulationScreen(
            simulation=_simulation,
            model=NeuralNetwork(
                _simulation.input_dim, [8, 8], _simulation.output_dim
            ).to("cuda"),
            ctx=self.ctx,
        )

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
        self._current_screen.handle_event()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False

    def _update(self) -> None:
        self._current_screen.update()
        pygame.display.update()

    def _draw(self) -> None:
        self.ctx.surface.fill(Colors.GREY)
        self._current_screen.draw(self.ctx)
