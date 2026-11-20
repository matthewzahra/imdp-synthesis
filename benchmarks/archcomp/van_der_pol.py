'''
Controlled stochastic Van der Pol oscillator (VP) of the ARCH-COMP stochastic models category, in the form with
additive control and additive Gaussian noise (ARCH-COMP22, Problem 5), with sampling time tau = 0.1:

    x1[k+1] = x1[k] + tau x2[k] + w1[k]
    x2[k+1] = x2[k] + tau ((1 - x1[k]^2) x2[k] - x1[k]) + u[k] + w2[k],   w ~ N(0, 0.2 I),   u in [-1, 1].

Specification: reach the target while staying in the safe set, over an unbounded horizon. The safe set is the
partition boundary, so leaving it counts as a failure. Ten inputs evenly spaced in [-1, 1] (as IntervalMDP.jl), and
cells of width 0.1, so that the target is aligned with the grid. The benchmark defines no initial state.

The SVMDP path needs a noise partition whose intervals are narrower than the cells (41 per dimension over the
truncated support of +-1.79): with 25, the bound at x0 of VanDerPol is 0.004 instead of 0.33. Memory: VanDerPol
needs about 4.5 GB (SVMDP) and 5.5 GB (IMDP).

The other ARCH-COMP variants are not included: the Büchi specification G(a) & GF(b), the continuous-time model, and
PRoTECT's verification variant, whose noise is uniform (not supported).
'''

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.archcomp.common import boxes, set_grid, stack
from benchmarks.dynamics import setmath
from benchmarks.dynamics.distributions import GaussianDistr
from core.rl.config import RLConfig


class _VanDerPol:
    def __init__(self, args, safe_set, number_per_dim, target):
        self.linear = False
        self.independent_state_dims = None
        self.independent_input_dims = None
        self.n, self.p = 2, 1
        self.tau = 0.1
        self.state_variables = ['x1', 'x2']
        self.wrap = jnp.array([False, False])
        self.plot_dimensions = [0, 1]
        self.lump = 1

        if args.noise_distr != 'gaussian':
            raise ValueError(f'{type(self).__name__} defines Gaussian noise; --noise_distr {args.noise_distr} is not supported.')
        self.noise = GaussianDistr(np.array([0.2, 0.2]))
        self.noise.set_partition_probs(num_cells=[41, 41])

        self.uMin, self.uMax, self.num_actions = [-1.0], [1.0], [10]
        set_grid(self, safe_set, number_per_dim)
        self.goal = boxes(target)
        self.critical = boxes()
        self.spec, self.horizon = 'reach_avoid', None
        self.x0 = np.array([2.05, 0.05])

        self.rl_config = RLConfig(inflation_rate=[(-3, 3), (-3, 3)])

    def step(self, state, action, noise):
        x1, x2 = state[0], state[1]
        return jnp.array([x1 + self.tau * x2,
                          x2 + self.tau * ((1 - x1 ** 2) * x2 - x1) + action[0]]) + noise

    @partial(jax.jit, static_argnums=(0))
    def step_set(self, state_min, state_max, action_min, action_max):
        action_min = jnp.maximum(action_min, self.uMin)
        action_max = jnp.minimum(action_max, self.uMax)
        x1_min, x2_min = state_min[0], state_min[1]
        x1_max, x2_max = state_max[0], state_max[1]

        x1_next = (x1_min + self.tau * x2_min, x1_max + self.tau * x2_max)

        # x2 + tau (1 - x1^2) x2 = x2 (1 + tau (1 - x1^2)): x2 occurs once, which keeps the bound tight
        sq_min, sq_max = setmath.square(x1_min, x1_max)
        factor = (1 + self.tau * (1 - sq_max), 1 + self.tau * (1 - sq_min))
        prod_min, prod_max = setmath.mult((x2_min, x2_max), factor)
        x2_next = (prod_min - self.tau * x1_max + action_min[0], prod_max - self.tau * x1_min + action_max[0])

        return stack(x1_next[0], x2_next[0]), stack(x1_next[1], x2_next[1])


class VanDerPol(_VanDerPol):
    '''
    IntervalMDP.jl's ARCH-COMP25 variant (from ArchCompStochasticModels.jl): safe set [-5, 5]^2, target
    [-1.2, -0.9] x [-2.9, -2.0]. Cells of width 0.1 (IntervalMDP.jl used 0.2, which does not align with the target).
    Sources: ARCH-COMP25 report (arXiv:2506.17602), Sec. "Benchmarking IntervalMDP.jl"; ArchCompStochasticModels.jl.
    '''

    def __init__(self, args):
        super().__init__(args, safe_set=[[-5, -5], [5, 5]], number_per_dim=[100, 100],
                         target=[[-1.2, -2.9], [-0.9, -2.0]])


class VanDerPol_SySCoRe(_VanDerPol):
    '''
    SySCoRe's ARCH-COMP25 variant: safe set [-3, 3]^2, target [-1.4, -0.7] x [-2.9, -2.0] (the scLTL specification
    p1 U p2 is reach-avoid). SySCoRe does not state its input grid; this uses the ten inputs of VanDerPol.
    Sources: ARCH-COMP25 report (arXiv:2506.17602), Sec. "Benchmarking SySCoRe".
    '''

    def __init__(self, args):
        super().__init__(args, safe_set=[[-3, -3], [3, 3]], number_per_dim=[60, 60],
                         target=[[-1.4, -2.9], [-0.7, -2.0]])
