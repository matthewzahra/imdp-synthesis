'''
Reduced autonomous vehicle (AV) benchmark of the ARCH-COMP stochastic models category (ARCH-COMP25, Sec. 4.3.2;
from the IMPaCT paper), a kinematic bicycle model with sampling time Ts = 0.1:

    x1[k+1] = x1[k] + Ts u1 cos(alpha + x3[k]) / cos(alpha) + w1[k]
    x2[k+1] = x2[k] + Ts u1 sin(alpha + x3[k]) / cos(alpha) + w2[k]
    x3[k+1] = x3[k] + Ts u1 tan(u2) + w3[k],         alpha = arctan(tan(u2) / 2),   w ~ N(0, 2/3 I),

with wheel velocity u1 in [-1, 4] and steering angle u2 in [-0.4, 0.4], on the state space [-5, 5]^2 x [-3.4, 3.4].
Specification: reach T = [-5.75, 0.25] x [-0.25, 5.75] x [-3.45, 3.45] while avoiding
A = [-5.75, 0.25] x [-0.75, -0.25] x [-3.45, 3.45], over an unbounded horizon.

The sets are given for a grid of points spaced 0.5 apart in x1 and x2 (as in IMPaCT). On the grid of cells used
here, the target shrinks to x1 <= 0, x2 >= 0, and the avoid set grows to x2 in [-1, 0], x1 <= 0.5 (both
conservative). The state space is the partition boundary, so leaving it counts as a failure; the orientation x3 is
not wrapped. The report defines no initial state; x0 lies right of the target and above the avoid set. The noise
(standard deviation 0.82 per step) is large relative to the largest step (0.4), so from states farther away the
vehicle mostly drifts out of the state space; and the heading of x0 is near 0, far from the bounds +-3.4 of the
orientation, which it leaves quickly otherwise.

Memory: the SVMDP path needs --dp_batch_size 100 (coarse grid, about 2 GB) or 50 (full grid, about 4 GB); with the
default batch size it exceeds 6 GB. The IMDP path needs about 4.3 GB for the coarse grid and over 6 GB for the full one. ArchCompStochasticModels.jl's version of these dynamics omits the "x +" and Ts terms; this follows the
report and IMPaCT's code (AV_minimal.cpp).
'''

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.archcomp.common import boxes, set_grid, stack
from benchmarks.dynamics import setmath
from benchmarks.dynamics.distributions import GaussianDistr
from core.rl.config import RLConfig


class _AutonomousVehicle3D:
    def __init__(self, args, number_per_dim, num_actions, noise_cells):
        self.n, self.p = 3, 2
        self.tau = 1
        self.Ts = 0.1
        self.state_variables = ['x', 'y', 'angle']
        self.wrap = jnp.array([False, False, False])
        self.plot_dimensions = [0, 1]

        if args.noise_distr != 'gaussian':
            raise ValueError(f'{type(self).__name__} defines Gaussian noise; --noise_distr {args.noise_distr} is not supported.')
        self.noise = GaussianDistr(np.array([2 / 3, 2 / 3, 2 / 3]))
        self.noise.set_partition_probs(num_cells=noise_cells)

        self.uMin, self.uMax, self.num_actions = [-1.0, -0.4], [4.0, 0.4], list(num_actions)
        set_grid(self, [[-5, -5, -3.4], [5, 5, 3.4]], number_per_dim)
        self.goal = boxes([[-5.75, -0.25, -3.45], [0.25, 5.75, 3.45]])
        self.critical = boxes([[-5.75, -0.75, -3.45], [0.25, -0.25, 3.45]])
        self.spec, self.horizon = 'reach_avoid', None
        self.x0 = np.array([0.75, 2.75, 0.1])

        self.rl_config = RLConfig(inflation_rate=[(-2, 2), (-2, 2), (-2, 2)])

    def step(self, state, action, noise):
        u1, u2 = action[0], action[1]
        alpha = jnp.arctan(jnp.tan(u2) / 2)
        speed = u1 / jnp.cos(alpha)
        return state + self.Ts * jnp.array([speed * jnp.cos(alpha + state[2]),
                                            speed * jnp.sin(alpha + state[2]),
                                            u1 * jnp.tan(u2)]) + noise

    @partial(jax.jit, static_argnums=(0))
    def step_set(self, state_min, state_max, action_min, action_max):
        action_min = jnp.maximum(action_min, self.uMin)
        action_max = jnp.minimum(action_max, self.uMax)
        u1 = (action_min[0], action_max[0])

        # |u2| <= 0.4 < pi/2, so tan(u2) and alpha increase with u2, and cos(alpha) > 0
        tan_u2 = (jnp.tan(action_min[1]), jnp.tan(action_max[1]))
        alpha = (jnp.arctan(tan_u2[0] / 2), jnp.arctan(tan_u2[1] / 2))
        cos_alpha_min, cos_alpha_max = setmath.cos(alpha[0], alpha[1])
        speed = setmath.mult(u1, (1 / cos_alpha_max, 1 / cos_alpha_min))

        heading = (alpha[0] + state_min[2], alpha[1] + state_max[2])
        dx = setmath.mult(speed, setmath.cos(*heading))
        dy = setmath.mult(speed, setmath.sin(*heading))
        dtheta = setmath.mult(u1, tan_u2)

        delta_min = self.Ts * stack(dx[0], dy[0], dtheta[0])
        delta_max = self.Ts * stack(dx[1], dy[1], dtheta[1])
        return state_min + delta_min, state_max + delta_max


class AutonomousVehicle3D(_AutonomousVehicle3D):
    '''
    The report's grid: cells of width (0.5, 0.5, 0.2) and inputs spaced (0.5, 0.1) apart (11 x 9 = 99 inputs).
    '''

    def __init__(self, args):
        super().__init__(args, number_per_dim=[20, 20, 34], num_actions=[11, 9], noise_cells=[7, 7, 5])


class AutonomousVehicle3D_coarse(_AutonomousVehicle3D):
    '''
    The report's reduced-complexity grid: cells of width (0.5, 0.5, 0.4) and inputs spaced (1, 0.2) apart
    (6 x 5 = 30 inputs). IMPaCT's ARCH-COMP25 entry used this input grid.
    '''

    def __init__(self, args):
        super().__init__(args, number_per_dim=[20, 20, 17], num_actions=[6, 5], noise_cells=[9, 9, 7])
