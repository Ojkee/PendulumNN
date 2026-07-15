from PendulumNN.ai import AIModel
from PendulumNN.common import Context
from PendulumNN.screens.screen import Screen
from PendulumNN.simulations import Simulation


class SimulationScreen(Screen):
    def __init__(self, simulation: Simulation, model: AIModel) -> None:
        super().__init__()
        self._simulation = simulation
        self.model = model

    def handle_event(self) -> None:
        pass

    def update(self) -> None:
        self._simulation.update()

    def draw(self, ctx: Context) -> None:
        self._simulation.draw(ctx)
