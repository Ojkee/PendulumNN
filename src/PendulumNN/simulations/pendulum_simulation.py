from itertools import pairwise
from dataclasses import dataclass
from functools import cache
import pygame
import torch
from PendulumNN.common import Colors, Context
from PendulumNN.simulations.simulation import Simulation

pos_t = tuple[int, int]


@dataclass
class _PendulumState:
    angles: torch.Tensor
    velocities: torch.Tensor
    cart_x: torch.Tensor
    cart_velocity: torch.Tensor

    def norm_cart_x(self, max_x: float) -> None:
        self.cart_x = self.cart_x / max_x

    def as_flat(self) -> torch.Tensor:
        return torch.cat(
            [
                self.angles,
                self.velocities,
                self.cart_x.reshape(1),
                self.cart_velocity.reshape(1),
            ],
        )


class PendulumSimulation(Simulation):
    LINE_WIDTH = 2
    CIRCLE_RADIUS = 4
    NODE_MASS = 2.0
    MAX_VELOCITIES = 25.0  # tested empirically
    CART_ACCELERATION = 10.0
    MAX_CAR_VELOCITY = 16.0  # tested empirically
    GRAVITY = 9.81

    def __init__(
        self,
        nodes: int,
        pendulum_length: int = 20,
        damping: float = 0.0,
        dt: float = 0.001,
    ) -> None:
        self._num_nodes = nodes
        self._pendulum_scale = pendulum_length
        self._damping_factor = damping
        self._dt = dt

        self._cart_x = torch.zeros((), requires_grad=False)
        self._cart_velocity = torch.zeros((), requires_grad=False)
        self._cart_acceleration = torch.zeros((), requires_grad=False)

        self._lens = torch.ones(self._num_nodes, requires_grad=False)
        self._masses = torch.ones(self._num_nodes, requires_grad=False) * self.NODE_MASS
        self._starting_angles = (
            torch.ones(self._num_nodes, requires_grad=False) * torch.pi
        )

        self._angles = self._starting_angles.clone()
        self._angles[-1] += (torch.rand(()) * 2 - 1) * 0.1

        self._velocities = torch.zeros(self._num_nodes, requires_grad=False)

        self._update_forces = (
            self._damping,
            self._cart_velocity_influence,
        )
        self.reset()

    def draw(self, ctx: Context) -> None:
        offset_x = ctx.width // 2
        offset_y = ctx.height // 4
        self._draw_axis(offset_x, offset_y, ctx.width, ctx.surface)
        self._draw_pendulum(offset_x, offset_y, ctx.surface)

    def _draw_pendulum(
        self,
        offset_x: int,
        offset_y: int,
        surface: pygame.Surface,
    ) -> None:
        cart_x = int(self._cart_x.item() * self._pendulum_scale) + offset_x
        origin = [(cart_x, offset_y)]

        xs = (self._xs() + offset_x).tolist()
        ys = (-self._ys() + offset_y).tolist()
        nodes = list(zip(xs, ys))

        for lhs, rhs in pairwise(origin + nodes):
            self._line(surface, lhs, rhs)
            self._point(surface, rhs)

    def _draw_axis(
        self,
        offset_x: int,
        offset_y: int,
        screen_width: int,
        surface: pygame.Surface,
    ) -> None:
        for h in range(offset_y - 100, offset_y + 100, 10):
            pygame.draw.circle(
                surface, Colors.LIGHT_GREY, (offset_x, h), radius=1, width=1
            )
        pygame.draw.line(
            surface, Colors.BEIGE, (0, offset_y), (screen_width, offset_y), width=1
        )

    @classmethod
    def _line(cls, surface: pygame.Surface, lhs: pos_t, rhs: pos_t) -> None:
        pygame.draw.line(surface, Colors.BEIGE, lhs, rhs, cls.LINE_WIDTH)

    @classmethod
    def _point(cls, surface: pygame.Surface, center: pos_t) -> None:
        pygame.draw.circle(
            surface, Colors.BEIGE, center, cls.CIRCLE_RADIUS, cls.CIRCLE_RADIUS
        )

    @classmethod
    def _text(cls, surface: pygame.Surface, text: str, pos: pos_t) -> None:
        font = pygame.font.SysFont("Comic Sans MS", 18)
        text_surface = font.render(text, False, Colors.BEIGE)
        surface.blit(text_surface, pos)

    @torch.no_grad()
    def update(self) -> None:
        self._euler_strategy()
        self._cart_acceleration = torch.zeros(())

    @torch.no_grad()
    def _euler_strategy(self) -> None:
        self._velocities += self.alpha * self._dt
        self._angles += self._velocities * self._dt
        self._cart_velocity += self._cart_acceleration * self._dt
        self._cart_velocity *= 1.0 - self._damping_factor * self._dt
        self._cart_x += self._cart_velocity * self._dt

    def _xs(self) -> torch.Tensor:
        positions = torch.cumsum(
            self._pendulum_scale * self._lens * torch.sin(self._angles),
            dim=0,
        )
        cart_offset = self._cart_x * self._pendulum_scale
        return (positions + cart_offset).round().to(torch.int64)

    def _ys(self) -> torch.Tensor:
        positions = torch.cumsum(
            -self._pendulum_scale * self._lens * torch.cos(self._angles),
            dim=0,
        )
        return positions.round().to(torch.int64)

    @property
    def alpha(self) -> torch.Tensor:
        with torch.no_grad():
            return torch.linalg.solve(self.M, self.f)

    @property
    def M(self) -> torch.Tensor:
        with torch.no_grad():
            _M = torch.zeros(self._num_nodes, self._num_nodes)
            for i in range(self._num_nodes):
                for j in range(self._num_nodes):
                    lhs = torch.sum(self._masses[max(i, j) :])
                    rhs = (
                        self._lens[i]
                        * self._lens[j]
                        * torch.cos(self._angles[i] - self._angles[j])
                    )
                    _M[i, j] = lhs * rhs
            return _M

    @property
    def f(self) -> torch.Tensor:
        with torch.no_grad():
            f = torch.zeros(self._num_nodes)
            grav = PendulumSimulation.GRAVITY * self._lens * torch.sin(self._angles)
            for i in range(self._num_nodes):
                lhs_masses = torch.sum(self._masses[i:])
                lhs = lhs_masses * grav[i]
                rhs_array = torch.zeros(self._num_nodes)
                for j in range(self._num_nodes):
                    k = max(i, j)
                    masses = torch.sum(self._masses[k:])
                    sin_diff = (
                        self._lens[i]
                        * self._lens[j]
                        * torch.pow(self._velocities[j], 2)
                        * torch.sin(self._angles[i] - self._angles[j])
                    )
                    rhs_array[j] = masses * sin_diff
                rhs = torch.sum(rhs_array)
                f[i] = -lhs - rhs
                for force in self._update_forces:
                    f[i] -= force(i)
            return f

    def _damping(self, i: int) -> torch.Tensor:
        return self._damping_factor * self._velocities[i]

    def _cart_velocity_influence(self, i: int) -> torch.Tensor:
        return (
            torch.sum(self._masses[i:])
            * self._lens[i]
            * torch.cos(self._angles[i])
            * self._cart_acceleration
        )

    @property
    def state(self) -> _PendulumState:
        return _PendulumState(
            angles=self._angles,
            velocities=self._velocities / PendulumSimulation.MAX_VELOCITIES,
            cart_x=self._cart_x,
            cart_velocity=self._cart_velocity / PendulumSimulation.MAX_CAR_VELOCITY,
        )

    @property
    @cache
    def input_dim(self) -> int:
        return len(self.state.as_flat())

    @property
    @cache
    def output_dim(self) -> int:
        return 3  # left, stay, right

    @property
    def input_state_vector(self) -> torch.Tensor:
        return self.state.as_flat()

    def handle_output_vector(self, y: torch.Tensor) -> None:
        match int(torch.argmax(y)):
            case 0:  # ACCELERATE LEFT
                self._cart_acceleration = torch.tensor(
                    -PendulumSimulation.CART_ACCELERATION
                )
            case 1:  # STAY
                pass
            case 2:  # ACCELERATE RIGHT
                self._cart_acceleration = torch.tensor(
                    PendulumSimulation.CART_ACCELERATION
                )

    def reward(self, ctx: Context) -> torch.Tensor:
        return -self.loss(ctx)

    def loss(self, ctx: Context) -> torch.Tensor:
        def _angle_diff(angle: torch.Tensor, target) -> torch.Tensor:
            diff = angle - target
            return torch.atan2(torch.sin(diff), torch.cos(diff))

        state = self.state
        state.norm_cart_x(ctx.width // 2)
        angle_loss = _angle_diff(state.angles, torch.pi).abs().sum()
        far_from_center = torch.abs(state.cart_x)
        return angle_loss + 0.01 * far_from_center

    @torch.no_grad()
    def reset(self) -> None:
        self._cart_x = torch.zeros(())
        self._cart_velocity = torch.zeros(())
        self._cart_acceleration = torch.zeros(())

        self._angles = self._starting_angles.clone()
        angle_noise: float = 0.05
        self._angles += (torch.rand(self._num_nodes) * 2 - 1) * angle_noise

        self._velocities = torch.zeros(self._num_nodes)
