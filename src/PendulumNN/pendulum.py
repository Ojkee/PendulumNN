from enum import Enum, auto
from itertools import pairwise
from typing import Callable

import pygame
import numpy as np

from PendulumNN.models import Colors

pos_t = tuple[int, int]
float_t = np.float64
strategy_fn = Callable[[], None]


class UpdateStrategy(Enum):
    EULER = auto()
    RK4 = auto()


class PendulumSimulation:
    LINE_WIDTH = 2
    CIRCLE_RADIUS = 4
    NODE_MASS = 2

    CART_ACCELERATION = float_t(10)
    GRAVITY = float_t(9.81)

    def __init__(
        self,
        x: int,
        y: int,
        nodes: int,
        pendulum_length: int = 20,
        damping: float = 0.0,
        dt: float = 0.001,
    ) -> None:
        self._offset_x = x
        self._offset_y = y
        self._num_nodes = nodes

        self._pendulum_scale = pendulum_length
        self._damping_factor = damping
        self._dt = dt

        self._cart_x = 0
        self._cart_velocity = float_t(0.0)
        self._cart_acceleration = float_t(0.0)

        self._lens = np.ones(self._num_nodes, dtype=float_t)
        self._masses = np.ones(self._num_nodes, dtype=float_t) * self.NODE_MASS
        self._angles = np.ones(self._num_nodes, dtype=float_t) * 0
        self._velocities = np.zeros(self._num_nodes, dtype=float_t)

        self._update_forces = (
            self._damping,
            self._cart_velocity_influence,
        )

        self._update_strategies: dict[UpdateStrategy, strategy_fn] = {
            UpdateStrategy.EULER: self._euler_strategy,
            UpdateStrategy.RK4: self._rk4_strategy,
        }

    def draw(self, screen: pygame.Surface) -> None:
        cart_x = int(self._cart_x * self._pendulum_scale) + self._offset_x
        origin = [(cart_x, self._offset_y)]
        nodes = list(zip(self._xs() + self._offset_x, -self._ys() + self._offset_y))
        for lhs, rhs in pairwise(origin + nodes):
            self._line(screen, lhs, rhs)
            self._point(screen, rhs)

    @classmethod
    def _line(cls, screen: pygame.Surface, lhs: pos_t, rhs: pos_t) -> None:
        pygame.draw.line(screen, Colors.BEIGE, lhs, rhs, cls.LINE_WIDTH)

    @classmethod
    def _point(cls, screen: pygame.Surface, center: pos_t) -> None:
        pygame.draw.circle(
            screen,
            Colors.BEIGE,
            center,
            cls.CIRCLE_RADIUS,
            cls.CIRCLE_RADIUS,
        )

    @classmethod
    def _text(cls, screen: pygame.Surface, text: str, pos: pos_t) -> None:
        font = pygame.font.SysFont("Comic Sans MS", 18)
        text_surface = font.render(text, False, Colors.BEIGE)
        screen.blit(text_surface, pos)

    def update(self) -> None:
        update_strategy = self._update_strategies[UpdateStrategy.EULER]
        update_strategy()

    def _euler_strategy(self) -> None:
        self._velocities += self.alpha * self._dt
        self._angles += self._velocities * self._dt

        self._cart_velocity += self._cart_acceleration * self._dt
        self._cart_velocity *= 1.0 - self._damping_factor * self._dt
        self._cart_x += self._cart_velocity * self._dt
        self._cart_acceleration = float_t(0)

    def _rk4_strategy(self) -> None:
        pass

    def _xs(self) -> np.ndarray:
        return np.cumsum(
            self._pendulum_scale * self._lens * np.sin(self._angles),
            dtype=np.int16,
        ) + np.int16(self._cart_x * self._pendulum_scale)

    def _ys(self) -> np.ndarray:
        return np.cumsum(
            -self._pendulum_scale * self._lens * np.cos(self._angles),
            dtype=np.int16,
        )

    @property
    def alpha(self) -> np.ndarray:
        return np.linalg.solve(self.M, self.f)

    @property
    def M(self) -> np.ndarray:
        # TODO: M is symmetric, therefore it can be optimized
        _M = np.zeros(shape=(self._num_nodes, self._num_nodes))
        for i in range(self._num_nodes):
            for j in range(self._num_nodes):
                lhs = np.sum(self._masses[max(i, j) :])
                rhs = (
                    self._lens[i]
                    * self._lens[j]
                    * np.cos(self._angles[i] - self._angles[j])
                )
                _M[i][j] = lhs * rhs
        return _M

    @property
    def f(self) -> np.ndarray:
        f = np.zeros(self._num_nodes)
        grav = PendulumSimulation.GRAVITY * self._lens * np.sin(self._angles)
        for i in range(self._num_nodes):
            lhs_masses = np.sum(self._masses[i:])
            lhs = lhs_masses * grav[i]
            rhs_array = np.zeros(self._num_nodes)
            for j in range(self._num_nodes):
                k = max(i, j)
                masses = np.sum(self._masses[k:])
                sin_diff = (
                    self._lens[i]
                    * self._lens[j]
                    * np.pow(self._velocities[j], 2)
                    * np.sin(self._angles[i] - self._angles[j])
                )
                rhs_array[j] = masses * sin_diff
            rhs = np.sum(rhs_array)
            f[i] = -lhs - rhs
            for force in self._update_forces:
                f[i] -= force(i)

        return f

    def _damping(self, i: int) -> float_t:
        return self._damping_factor * self._velocities[i]

    def _cart_velocity_influence(self, i: int) -> float_t:
        return (
            np.sum(self._masses[i:])
            * self._lens[i]
            * np.cos(self._angles[i])
            * self._cart_acceleration
        )

    def accelerate_left(self) -> None:
        self._accelerate_horizontal(-PendulumSimulation.CART_ACCELERATION)

    def accelerate_right(self) -> None:
        self._accelerate_horizontal(PendulumSimulation.CART_ACCELERATION)

    def _accelerate_horizontal(self, da: float) -> None:
        self._cart_acceleration = da
