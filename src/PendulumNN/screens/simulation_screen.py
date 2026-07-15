import pygame
import torch

from PendulumNN.ai import AIModel
from PendulumNN.common import Context
from PendulumNN.screens.screen import Screen
from PendulumNN.simulations import Simulation


class SimulationScreen(Screen):
    def __init__(self, simulation: Simulation, model: AIModel, ctx: Context) -> None:
        super().__init__()
        self._simulation = simulation
        self._model = model
        self._ctx = ctx

    def handle_event(self) -> None:
        keys = pygame.key.get_pressed()
        if keys[pygame.K_r]:
            self._simulation.reset()

        # manual
        if keys[pygame.K_a]:
            self._simulation.handle_output_vector(torch.tensor([1, 0, 0]))
        elif keys[pygame.K_d]:
            self._simulation.handle_output_vector(torch.tensor([0, 0, 1]))

    def update(self) -> None:
        self._simulation.update()

    def draw(self, ctx: Context) -> None:
        self._simulation.draw(ctx)
