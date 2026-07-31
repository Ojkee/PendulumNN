import pygame
import torch

from torch.distributions import Categorical
import torch.nn.functional as F

from PendulumNN.ai import AIModel
from PendulumNN.common import Context
from PendulumNN.screens import Screen
from PendulumNN.simulations import Simulation


class SimulationScreen(Screen):
    def __init__(
        self,
        simulation: Simulation,
        model: AIModel,
        ctx: Context,
    ) -> None:
        super().__init__()
        self._simulation = simulation
        self._model = model
        self._ctx = ctx

        self.training = False

        self._simulation.reset()

    def handle_event(self) -> None:
        state = self._simulation.input_state_vector
        logits = self._model(state.to("cuda"))
        dist = Categorical(logits=logits)
        action = dist.sample()
        self._simulation.handle_output_vector(
            F.one_hot(action, self._simulation.output_dim)
        )

        self._user_control()

    def update(self) -> None:
        if self.training:
            self.train_model(epochs=100)
            self.switch_train()

        self._simulation.update()

    def draw(self, ctx: Context) -> None:
        self._simulation.draw(ctx)
        self._model.draw(ctx)

    def _user_control(self) -> None:
        keys = pygame.key.get_pressed()
        if keys[pygame.K_r]:
            self.switch_train()

    def train_model(self, epochs: int) -> None:
        SIM_STEPS = 500
        GAMMA = 0.99
        for epoch in range(epochs):
            self._simulation.reset()

            log_probs, rewards = self._one_simulation(SIM_STEPS)
            episode_return = torch.stack(rewards).sum().item()  # <- TO loguj

            returns = []
            G = torch.tensor(0.0)
            for r in reversed(rewards):
                G = r + GAMMA * G
                returns.insert(0, G)
            returns = torch.stack(returns)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            loss = -torch.sum(torch.stack(log_probs) * returns.to("cuda"))
            self._model.update(loss)

            episode_return = torch.stack(rewards).sum().item()
            mean_step_loss = -episode_return / len(rewards)
            if epoch % 50 == 0:
                print(
                    f"{epoch:>3}/{epochs}   "
                    f"surrogate_loss={loss.item():.4f}   "
                    f"mean_step_loss={mean_step_loss:.4f}"
                )

    def _one_simulation(
        self, steps: int
    ) -> tuple[list[torch.Tensor], list[torch.torch.Tensor]]:
        log_probs = []
        rewards = []
        for _ in range(steps):
            state = self._simulation.input_state_vector
            logits = self._model(state.to("cuda"))
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            self._simulation.handle_output_vector(
                F.one_hot(action, self._simulation.output_dim)
            )
            self._simulation.update()
            rewards.append(self._simulation.reward(self._ctx))
        return log_probs, rewards

    def switch_train(self):
        self.training = not self.training
        self._simulation.reset()
