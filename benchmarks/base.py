import jax.numpy as jnp
import numpy as np

from benchmarks.dynamics.distributions import GaussianDistr, TriangularDistr
from core.box import Box
from core.specification import REACH_AVOID


class Benchmark:
    '''
    Base class of the benchmarks: a discrete-time system x' = f(x, u) + w with additive noise w, and a specification.

    The dynamics (benchmarks/models.py) define:
        state_variables: name per state dimension
        step(x, u, w): the successor state
        step_set(x_min, x_max, u_min, u_max): bounds on the noise-free successors of a box of states and inputs
        noise: the distribution of w, set with set_noise()
        wrapped: names of the state dimensions that wrap around (angles), if any

    Each benchmark then defines:
        state_space: Box with names=state_variables; leaving it counts as a failure (for safety, it is the safe set)
        state_grid: number of partition cells per state dimension
        input_space: Box of the control inputs; input_grid: number of discrete inputs per input dimension
        goal, critical: lists of Boxes in the state space (usually built with state_space.restrict)
        x0: initial state
        plot_dimensions: the two state dimensions to plot
    and optionally spec and horizon (see core/specification.py), rl_config (core.rl.config.RLConfig), and
    plot_simulation() for plots of the simulated traces of its own (such as an animation). create_model() calls
    check() on the finished benchmark.
    '''

    spec = REACH_AVOID
    horizon = None
    rl_config = None
    wrapped = ()
    goal = ()
    critical = ()

    @property
    def n(self):
        '''Number of state dimensions.'''
        return self.state_space.dim

    @property
    def p(self):
        '''Number of input dimensions.'''
        return self.input_space.dim

    @property
    def wrap(self):
        '''Per state dimension, whether it wraps around (bool array of shape [n]).'''
        return jnp.array([variable in self.wrapped for variable in self.state_variables], dtype=bool)

    def set_noise(self, args, gaussian=None, triangular=None):
        '''
        Set the noise to the distribution selected with --noise_distr.

        :param args: Argument namespace (provides noise_distr)
        :param gaussian: dict(variance=..., cells=...): the variance per state dimension and the number of cells per
            dimension of the noise partition; None if the benchmark does not define Gaussian noise
        :param triangular: dict(halfwidth=..., cells=...); None if the benchmark does not define triangular noise
        '''
        if args.noise_distr == 'gaussian' and gaussian is not None:
            self.noise = GaussianDistr(np.asarray(gaussian['variance'], dtype=float))
            cells = gaussian['cells']
        elif args.noise_distr == 'triangular' and triangular is not None:
            self.noise = TriangularDistr(np.asarray(triangular['halfwidth'], dtype=float))
            cells = triangular['cells']
        else:
            supported = ' or '.join(name for name, params in (('Gaussian', gaussian), ('triangular', triangular))
                                    if params is not None)
            raise ValueError(f'{type(self).__name__} defines {supported} noise; '
                             f'--noise_distr {args.noise_distr} is not supported.')
        self.noise.set_partition_probs(num_cells=cells)

    def plot_simulation(self, args, stamp, partition, traces):
        '''
        Plots of the Monte Carlo traces that are specific to the benchmark (none by default).

        :param args: Argument namespace (provides output_dir)
        :param stamp: Time stamp of the run, for file names
        :param partition: Partition of the state space
        :param traces: Simulated traces, {run: {'k', 'x', 'u'}}
        '''

    def check(self):
        '''Check that the benchmark is complete and consistent; raises a ValueError otherwise.'''
        name = type(self).__name__

        for attr in ('state_space', 'input_space'):
            if not isinstance(getattr(self, attr, None), Box):
                raise ValueError(f'{name}: {attr} must be a Box.')
        if self.state_space.names != tuple(self.state_variables):
            raise ValueError(f'{name}: the state space must be named after the state variables '
                             f'{list(self.state_variables)}, got {self.state_space.names}.')

        for attr, dim in (('state_grid', self.n), ('input_grid', self.p)):
            grid = np.asarray(getattr(self, attr))
            if grid.shape != (dim,) or np.any(grid < 1) or not np.all(grid == np.round(grid)):
                raise ValueError(f'{name}: {attr} must hold a positive integer for each of the {dim} dimensions, '
                                 f'got {grid.tolist()}.')

        for attr in ('goal', 'critical', 'charging_station'):
            for box in getattr(self, attr, ()):
                if not isinstance(box, Box) or box.dim != self.n:
                    raise ValueError(f'{name}: {attr} must be a list of Boxes of dimension {self.n}, got {box!r}.')

        x0 = np.asarray(self.x0, dtype=float)
        if x0.shape != (self.n,) or not self.state_space.contains(x0):
            raise ValueError(f'{name}: x0 = {x0} must be a point in the state space {self.state_space}.')
        if len(self.noise['mean']) != self.n:
            raise ValueError(f"{name}: the noise has {len(self.noise['mean'])} dimensions, the state {self.n}.")
        if not all(0 <= d < self.n for d in self.plot_dimensions):
            raise ValueError(f'{name}: plot_dimensions {self.plot_dimensions} must be state dimensions.')
