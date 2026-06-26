from __future__ import annotations
import torch
import pygame
from PendulumNN.models import Colors
from PendulumNN.neural_net import NeuralNetwork, Prediction
from PendulumNN.pendulum import PendulumSimulation, UpdateStrategy, Stats


class Window:
    EPISODE_LENGTH = 1000
    LEARNING_RATE = 1e-3
    GAMMA = 0.99  # discount factor for returns

    def __init__(self, width: int, height: int) -> None:
        self._width = width
        self._height = height
        self._running: bool = True
        self._screen: pygame.Surface | None = None
        self._pendulum = PendulumSimulation(
            self._width // 2,
            self._height // 2,
            nodes=1,
            pendulum_length=50,
            damping=0.5,
            update_strategy=UpdateStrategy.EULER,
        )
        self.model = NeuralNetwork(
            input_dim=len(self._pendulum.stats.as_flat()),
            hidden_layers=2,
        ).to(NeuralNetwork.DEVICE)
        self._optimizer = torch.optim.Adam(  # type: ignore
            self.model.parameters(), lr=self.LEARNING_RATE
        )
        self._step_count = 0
        self._episode_count = 0
        self._log_probs: list[torch.Tensor] = []
        self._rewards: list[float] = []

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
            self._nn_control()
            self._draw()
            self._step_count += 1
            if self._step_count >= self.EPISODE_LENGTH:
                self._end_episode()

    def _check_event(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False

    def _update(self) -> None:
        self._pendulum.update()

    def _nn_control(self) -> None:
        stats: Stats = self._pendulum.stats
        stats.norm_cart_x(self._width / 2)

        action, log_prob = self.model.act(stats.as_flat())
        self._apply_action(action)

        reward = -float(self.model.loss(stats))
        self._log_probs.append(log_prob)
        self._rewards.append(reward)

    def _apply_action(self, action: Prediction) -> None:
        match action:
            case Prediction.LEFT:
                self._pendulum.accelerate_left()
            case Prediction.RIGHT:
                self._pendulum.accelerate_right()
            case Prediction.STAY:
                pass

    def _end_episode(self) -> None:
        returns = self._compute_returns()
        self._reinforce_update(returns)

        self._episode_count += 1
        print(
            f"Episode {self._episode_count} " f"total_reward={sum(self._rewards):.2f}"
        )

        self._log_probs.clear()
        self._rewards.clear()
        self._step_count = 0
        self._pendulum.reset()

    def _compute_returns(self) -> torch.Tensor:
        returns = torch.zeros(len(self._rewards), dtype=torch.float32)
        running_return = 0.0
        for t in reversed(range(len(self._rewards))):
            running_return = self._rewards[t] + self.GAMMA * running_return
            returns[t] = running_return
        # normalize for training stability
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns.to(NeuralNetwork.DEVICE)

    def _reinforce_update(self, returns: torch.Tensor) -> None:
        log_probs = torch.stack(self._log_probs)
        policy_loss = -(log_probs * returns).sum()

        self._optimizer.zero_grad()
        policy_loss.backward()
        self._optimizer.step()

    def _draw(self) -> None:
        self.screen.fill(Colors.GREY)
        self._draw_center_dot_line()
        self._pendulum.draw(self.screen)
        pygame.display.update()

    def _draw_center_dot_line(self) -> None:
        for h in range(100, self._height // 2 + 100, 10):
            pygame.draw.circle(
                self.screen,
                (*Colors.BEIGE, 24),
                (self._width // 2, h),
                radius=1,
                width=1,
            )
        pygame.draw.line(
            self.screen,
            (*Colors.BEIGE, 24),
            (0, self._height // 2),
            (self._width, self._height // 2),
            width=1,
        )
