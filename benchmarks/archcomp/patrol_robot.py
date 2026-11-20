'''
Reduced patrol robot (PR) benchmark of the ARCH-COMP stochastic models category (ARCH-COMP25, Sec. 4.3.1; taken from
the IMPaCT paper):

    x[k+1] = x[k] + 10 u1[k] (cos u2[k], sin u2[k]) + w[k],   w ~ N(0, 0.75 I),   u in [-1, 1]^2,

on the state space X = [-10, 10]^2, with target T = [5, 7]^2 and avoid set A = [-2, 2]^2, over an unbounded horizon.
The state space is the partition boundary, so leaving it counts as a failure. The report defines no initial state;
x0 lies in the opposite corner from the target, so the robot must move around the avoid set.

The variants with the bounded disturbance w in [-0.5, 0.5] are not included (they need the disturbance to be
covered in the bounds and sampled in the simulator). IMPaCT's ARCH-COMP25 code (PR_minimal.cpp) differs from the
report: step 2 u1 instead of 10 u1, target [5, 8]^2, and avoid set [-1.5, 1.5]^2. These benchmarks follow the report.
'''

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.archcomp.common import boxes, set_grid, stack
from benchmarks.dynamics import setmath
from benchmarks.dynamics.distributions import GaussianDistr
from core.rl.config import RLConfig


class _PatrolRobot:
    def __init__(self, args, number_per_dim, num_actions, noise_cells):
        self.linear = False
        self.independent_state_dims = None
        self.independent_input_dims = None
        self.n, self.p = 2, 2
        self.tau = 1
        self.state_variables = ['x', 'y']
        self.wrap = jnp.array([False, False])
        self.plot_dimensions = [0, 1]
        self.lump = 1
        self.step_size = 10.0

        if args.noise_distr != 'gaussian':
            raise ValueError(f'{type(self).__name__} defines Gaussian noise; --noise_distr {args.noise_distr} is not supported.')
        self.noise = GaussianDistr(np.array([0.75, 0.75]))
        self.noise.set_partition_probs(num_cells=noise_cells)

        self.uMin, self.uMax, self.num_actions = [-1.0, -1.0], [1.0, 1.0], list(num_actions)
        set_grid(self, [[-10, -10], [10, 10]], number_per_dim)
        self.x0 = np.array([-7.25, -7.25])

    def step(self, state, action, noise):
        move = self.step_size * action[0] * jnp.array([jnp.cos(action[1]), jnp.sin(action[1])])
        return state + move + noise

    @partial(jax.jit, static_argnums=(0))
    def step_set(self, state_min, state_max, action_min, action_max):
        action_min = jnp.maximum(action_min, self.uMin)
        action_max = jnp.minimum(action_max, self.uMax)
        speed = (self.step_size * action_min[0], self.step_size * action_max[0])
        dx_min, dx_max = setmath.mult(speed, setmath.cos(action_min[1], action_max[1]))
        dy_min, dy_max = setmath.mult(speed, setmath.sin(action_min[1], action_max[1]))
        return state_min + stack(dx_min, dy_min), state_max + stack(dx_max, dy_max)


class PatrolRobot_reachavoid(_PatrolRobot):
    '''
    Reach T while avoiding A (the report's first scenario), with the report's grid: cells of width 0.5 and inputs
    spaced 0.1 apart (21 x 21 = 441 inputs). Memory: about 4.5 GB (SVMDP) and 5.3 GB (IMDP).
    '''

    def __init__(self, args):
        super().__init__(args, number_per_dim=[40, 40], num_actions=[21, 21], noise_cells=[15, 15])
        self.goal = boxes([[5, 5], [7, 7]])
        self.critical = boxes([[-2, -2], [2, 2]])
        self.spec, self.horizon = 'reach_avoid', None
        self.rl_config = RLConfig(inflation_rate=[(-2, 2), (-2, 2)])


class PatrolRobot_reach(_PatrolRobot):
    '''
    Reach T, without the avoid set (the report's third scenario, intended for fast runs), with the report's coarser
    grid: cells of width 1 and inputs spaced 0.2 apart (11 x 11 = 121 inputs).
    '''

    def __init__(self, args):
        super().__init__(args, number_per_dim=[20, 20], num_actions=[11, 11], noise_cells=[21, 21])
        self.goal = boxes([[5, 5], [7, 7]])
        self.critical = boxes()
        self.spec, self.horizon = 'reach_avoid', None
        self.rl_config = RLConfig(inflation_rate=[(-2, 2), (-2, 2)])
