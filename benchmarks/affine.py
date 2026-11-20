'''
Affine dynamics x' = A x + B u + q + w, and the benchmarks built on them: DoubleIntegrator and Test1D (the ARCH-COMP
benchmarks in benchmarks/archcomp/ use them as well).
'''

import jax.numpy as jnp
import numpy as np

from benchmarks.base import Benchmark
from core.box import Box
from core.rl.config import RLConfig


class AffineDynamics(Benchmark):
    '''
    Affine dynamics with additive noise: x' = A x + B u + q + w.

    Benchmarks call AffineDynamics.__init__ with their matrices and noise (see Benchmark.set_noise) and then define
    their specification.
    '''

    def __init__(self, args, A, B, q, gaussian=None, triangular=None, state_variables=None):
        '''
        :param args: Argument namespace (provides noise_distr)
        :param A: State matrix, shape [n, n]
        :param B: Input matrix, shape [n, p]
        :param q: Constant drift, shape [n]
        :param gaussian: Gaussian noise, dict(variance=..., cells=...); None if not supported
        :param triangular: Triangular noise, dict(halfwidth=..., cells=...); None if not supported
        :param state_variables: Names of the state variables (default x1, ..., xn)
        '''
        self.A = np.asarray(A, dtype=float)
        self.B = np.asarray(B, dtype=float)
        self.q = np.asarray(q, dtype=float).reshape(-1)
        self.tau = 1
        n = self.A.shape[0]
        self.state_variables = list(state_variables) if state_variables is not None else [f'x{i + 1}' for i in range(n)]
        self.set_noise(args, gaussian=gaussian, triangular=triangular)

    def step(self, state, action, noise):
        return jnp.dot(self.A, state) + jnp.dot(self.B, action) + self.q + noise

    def successor_bounds(self, state_min, state_max, action_min, action_max):

        # The image of a box under an affine map is bounded exactly by its center +- |M| times its radius
        state_center, state_radius = (state_min + state_max) / 2, (state_max - state_min) / 2
        action_center, action_radius = (action_min + action_max) / 2, (action_max - action_min) / 2

        center = jnp.dot(self.A, state_center) + jnp.dot(self.B, action_center) + self.q
        radius = jnp.dot(np.abs(self.A), state_radius) + jnp.dot(np.abs(self.B), action_radius)

        return center - radius, center + radius


class DoubleIntegrator(AffineDynamics):
    '''
    Double integrator benchmark, with a 2D state space (position, velocity) and a 1D control input space.
    '''

    def __init__(self, args):
        tau = 1.0
        A = np.array([[1, tau],
                      [0, 1]])
        B = np.array([[tau**2 / 2],
                      [tau]])

        AffineDynamics.__init__(self, args, A=A, B=B, q=[0, 0],
                                gaussian=dict(variance=np.array([0.15, 0.15])**2, cells=[10, 10]),
                                triangular=dict(halfwidth=[0.15, 0.15], cells=[10, 10]),
                                state_variables=['position', 'velocity'])

        self.plot_dimensions = [0, 1]

        self.state_space = Box([-21, -10.5], [21, 10.5], names=self.state_variables)
        self.state_grid = [21, 21]
        self.input_space = Box([-5], [5])
        self.input_grid = [21]

        self.goal = [self.state_space.restrict(position=(-4, 4), velocity=(-2, 2))]
        self.critical = []

        self.x0 = np.array([0, -8])

        # RL configuration: networks, PPO training, reward function, and the tube
        # grown around the RL rollouts to form the abstraction.
        self.rl_config = RLConfig(
            pi_arch=[64, 64],
            vf_arch=[64, 64],
            goal_reward=5,
            unsafe_penalty=-5,
            out_of_bounds_penalty=-5,
            per_step_cost=0.1,
            distance_cost=0.0,
            inflation_rate=[(-2, 2), (-1, 1)],
        )


class Test1D(AffineDynamics):
    '''
    Simple 1D test benchmark (x' = x + u + w), with a 1D state space and a 1D control input space.
    '''

    def __init__(self, args):
        AffineDynamics.__init__(self, args, A=[[1]], B=[[1]], q=[0],
                                gaussian=dict(variance=np.array([0.2])**2, cells=[10]),
                                triangular=dict(halfwidth=[0.2], cells=[10]),
                                state_variables=['position'])

        self.plot_dimensions = [0]

        self.state_space = Box([-5], [5], names=self.state_variables)
        self.state_grid = [10]
        self.input_space = Box([-1], [1])
        self.input_grid = [3]

        self.goal = [self.state_space.restrict(position=(-1, 1))]
        self.critical = []

        self.x0 = np.array([-2.5])

        # RL configuration: networks, PPO training, reward function, and the tube
        # grown around the RL rollouts to form the abstraction.
        self.rl_config = RLConfig(
            pi_arch=[64, 64],
            vf_arch=[64, 64],
            goal_reward=5,
            unsafe_penalty=-5,
            out_of_bounds_penalty=-5,
            per_step_cost=0.1,
            distance_cost=0.0,
            inflation_rate=[(-1, 1)],
        )
