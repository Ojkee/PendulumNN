from enum import Enum, auto
from itertools import pairwise
from typing import Callable
from dataclasses import dataclass

import pygame
import numpy as np

from PendulumNN.common import Action, Colors

pos_t = tuple[int, int]
float_t = np.float64
strategy_fn = Callable[[], None]


class UpdateStrategy(Enum):
    EULER = auto()
    RK4 = auto()


@dataclass
class Stats:
    sin_angles: np.ndarray
    cos_angles: np.ndarray
    velocities: np.ndarray
    cart_x: float
    cart_velocity: float

    def norm_cart_x(self, max_x: float) -> None:
        self.cart_x /= max_x

    def as_flat(self) -> np.ndarray:
        return np.concatenate(
            [
                self.sin_angles,
                self.cos_angles,
                self.velocities,
                [self.cart_x, self.cart_velocity],
            ],
            dtype=float_t,
        )


class PendulumSimulation:
    LINE_WIDTH = 2
    CIRCLE_RADIUS = 4
    NODE_MASS = float_t(2.0)
    MAX_VELOCITIES = float_t(25.0)  # tested empirically

    CART_ACCELERATION = float_t(10.0)
    MAX_CAR_VELOCITY = float_t(16.0)  # tested empirically
    GRAVITY = float_t(9.81)

    def __init__(
        self,
        x: int,
        y: int,
        nodes: int,
        pendulum_length: int = 20,
        damping: float = 0.0,
        dt: float = 0.001,
        update_strategy: UpdateStrategy = UpdateStrategy.EULER,
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
        self._angles = np.ones(self._num_nodes, dtype=float_t) * np.pi
        self._angles[-1] += np.random.uniform(-0.1, 0.1)
        self._velocities = np.zeros(self._num_nodes, dtype=float_t)

        self._update_forces = (
            self._damping,
            self._cart_velocity_influence,
        )

        self._update_strategy_type = update_strategy
        self._update_strategies: dict[UpdateStrategy, strategy_fn] = {
            UpdateStrategy.EULER: self._euler_strategy,
            UpdateStrategy.RK4: self._rk4_strategy,
        }

        self.reset()

    def draw(self, surface: pygame.Surface) -> None:
        cart_x = int(self._cart_x * self._pendulum_scale) + self._offset_x
        origin = [(cart_x, self._offset_y)]
        nodes = list(zip(self._xs() + self._offset_x, -self._ys() + self._offset_y))
        for lhs, rhs in pairwise(origin + nodes):
            self._line(surface, lhs, rhs)
            self._point(surface, rhs)

    @classmethod
    def _line(cls, surface: pygame.Surface, lhs: pos_t, rhs: pos_t) -> None:
        pygame.draw.line(surface, Colors.BEIGE, lhs, rhs, cls.LINE_WIDTH)

    @classmethod
    def _point(cls, surface: pygame.Surface, center: pos_t) -> None:
        pygame.draw.circle(
            surface,
            Colors.BEIGE,
            center,
            cls.CIRCLE_RADIUS,
            cls.CIRCLE_RADIUS,
        )

    @classmethod
    def _text(cls, surface: pygame.Surface, text: str, pos: pos_t) -> None:
        font = pygame.font.SysFont("Comic Sans MS", 18)
        text_surface = font.render(text, False, Colors.BEIGE)
        surface.blit(text_surface, pos)

    def update(self) -> None:
        self.update_strategy()
        self._cart_acceleration = float_t(0)

    def _euler_strategy(self) -> None:
        self._velocities += self.alpha * self._dt
        self._angles += self._velocities * self._dt

        self._cart_velocity += self._cart_acceleration * self._dt
        self._cart_velocity *= 1.0 - self._damping_factor * self._dt
        self._cart_x += self._cart_velocity * self._dt

    def _rk4_strategy(self) -> None:
        a, v = self._angles.copy(), self._velocities.copy()
        x, cv = self._cart_x, self._cart_velocity
        ca = self._cart_acceleration

        dt = self._dt

        dv1, da1, dx1, dcv1 = self._derivatives(a, v, x, cv, ca)
        dv2, da2, dx2, dcv2 = self._derivatives(
            a + dt / 2 * da1, v + dt / 2 * dv1, x + dt / 2 * dx1, cv + dt / 2 * dcv1, ca
        )
        dv3, da3, dx3, dcv3 = self._derivatives(
            a + dt / 2 * da2, v + dt / 2 * dv2, x + dt / 2 * dx2, cv + dt / 2 * dcv2, ca
        )
        dv4, da4, dx4, dcv4 = self._derivatives(
            a + dt * da3, v + dt * dv3, x + dt * dx3, cv + dt * dcv3, ca
        )

        self._angles += dt / 6 * (da1 + 2 * da2 + 2 * da3 + da4)
        self._velocities += dt / 6 * (dv1 + 2 * dv2 + 2 * dv3 + dv4)
        self._cart_x += dt / 6 * (dx1 + 2 * dx2 + 2 * dx3 + dx4)
        self._cart_velocity += dt / 6 * (dcv1 + 2 * dcv2 + 2 * dcv3 + dcv4)

    def _derivatives(self, angles, velocities, cart_x, cart_vel, cart_acc):
        old_a, old_v, old_x, old_cv = (
            self._angles,
            self._velocities,
            self._cart_x,
            self._cart_velocity,
        )
        self._angles, self._velocities, self._cart_x, self._cart_velocity = (
            angles,
            velocities,
            cart_x,
            cart_vel,
        )
        self._cart_acceleration = cart_acc

        alpha = self.alpha  # ← tutaj, przed przywróceniem
        friction = self._damping_factor * cart_vel

        self._angles, self._velocities, self._cart_x, self._cart_velocity = (
            old_a,
            old_v,
            old_x,
            old_cv,
        )

        return velocities, alpha, cart_vel, cart_acc - friction

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

    @property
    def update_strategy(self) -> strategy_fn:
        return self._update_strategies[self._update_strategy_type]

    def _damping(self, i: int) -> float_t:
        return self._damping_factor * self._velocities[i]

    def _cart_velocity_influence(self, i: int) -> float_t:
        return (
            np.sum(self._masses[i:])
            * self._lens[i]
            * np.cos(self._angles[i])
            * self._cart_acceleration
        )

    def apply_action(self, action: Action) -> None:
        match action:
            case Action.LEFT:
                self._cart_acceleration = -PendulumSimulation.CART_ACCELERATION
            case Action.RIGHT:
                self._cart_acceleration = PendulumSimulation.CART_ACCELERATION

    @property
    def stats(self) -> Stats:
        return Stats(
            sin_angles=np.sin(self._angles),
            cos_angles=np.cos(self._angles),
            velocities=self._velocities / PendulumSimulation.MAX_VELOCITIES,
            cart_x=self._cart_x,
            cart_velocity=self._cart_velocity / PendulumSimulation.MAX_CAR_VELOCITY,
        )

    def reset(self, angle_noise: float = 0.05) -> None:
        rng = np.random.default_rng()
        self._cart_x = 0
        self._cart_velocity = float_t(0.0)
        self._cart_acceleration = float_t(0.0)
        self._angles = np.ones(self._num_nodes, dtype=float_t) * np.pi
        self._angles += rng.uniform(-angle_noise, angle_noise, size=self._num_nodes)
        self._velocities = np.zeros(self._num_nodes, dtype=float_t)
