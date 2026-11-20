import logging
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)

# Why a run ended, in the order the checks are made at each step
ABSORBING, GOAL, CRITICAL, HORIZON, NO_ACTION = range(5)
OUTCOMES = ('absorbing state reached', 'goal reached', 'critical state reached', 'horizon completed',
            'no enabled action')


class MonteCarloSim():
    '''
    Class to run Monte Carlo simulations on the discrete-time stochastic system closed under a fixed Markov policy.

    A run of a reach-avoid specification succeeds if it reaches the goal within sim_horizon steps, without a
    critical or absorbing state before. A run of a safety specification succeeds if it completes sim_horizon
    steps without a critical or absorbing state. A policy of shape [N, nr_states] is time-dependent (finite
    horizon); one of shape [nr_states] is stationary.

    All runs are simulated together: one jitted JAX loop advances every unfinished run by one step, until all
    runs have ended.
    '''

    def __init__(self, model, partition, policy, policy_inputs, x0, iterations=1000, sim_horizon=1000, random_initial_state=False, verbose=True, spec='reach_avoid', **kwargs):

        logger.info('=== Start Monte Carlo simulations ===')

        self.verbose = verbose

        self.model = model
        self.partition = partition

        self.policy = np.asarray(policy)
        self.policy_inputs = np.asarray(policy_inputs)
        self.horizon = sim_horizon
        self.spec = spec
        self.iterations = iterations
        self.random_initial_state = random_initial_state

        # Predefine noise to speed up computations
        self.define_noise_values()

        # Goal and critical membership per state ID, with a last (False) entry for the absorbing state
        self.is_goal = np.append(np.asarray(partition.goal['bools'], dtype=bool)[:partition.size], False)
        self.is_critical = np.append(np.asarray(partition.critical['bools'], dtype=bool)[:partition.size], False)

        self.results = {
            'satprob': -1,
            'goal_reached': np.full(self.iterations, False, dtype=bool),
            'traces': {}
        }

        # Simulate all runs at once
        outcome, length, x, u = self._runIterations(*self.initial_states(x0))

        self.results['goal_reached'] = (outcome == GOAL) | ((outcome == HORIZON) & (self.spec == 'safety'))
        self.results['satprob'] = np.mean(self.results['goal_reached'])

        # Trace of run m: the states x_0, ..., x_L and inputs u_0, ..., u_{L-1}, where L is the step at which it ended
        for m, L in enumerate(length.tolist()):
            self.results['traces'][m] = {'k': np.arange(L + 1), 'x': x[m, :L + 1], 'u': u[m, :L]}

        self.log_outcomes(outcome, length, x)

    def define_noise_values(self):
        '''
        Predefine the noise values to speed up computations.
        '''

        if hasattr(self.model.noise, 'sample'):
            self.noise = self.model.noise.sample(size=(self.iterations, self.horizon))
            return

        # Gaussian noise mode
        self.noise = np.random.multivariate_normal(
            self.model.noise['mean'], self.model.noise['cov'],
            (self.iterations, self.horizon))

    def initial_states(self, x0):
        '''
        Initial continuous state of each run: x0, or a uniform sample from the partition cell that contains x0.

        :return:
            - x0: Initial continuous state per run, shape [iterations, n].
            - s0: The state ID of the partition cell containing x0 (the cell's ID is also used for random initial
              states, which are sampled inside it).
        '''

        # Determine the initial region on the host with partition.x2state, as for the value in x0 that is reported.
        # x0 often lies on a cell boundary, where the rounding of the batched lookup could pick a neighboring cell.
        s0, _ = self.partition.x2state(x0)

        if self.random_initial_state:
            x0 = np.random.uniform(
                low=np.asarray(self.partition.regions['lower_bounds'][s0]),
                high=np.asarray(self.partition.regions['upper_bounds'][s0]),
                size=(self.iterations, self.model.n))
        else:
            x0 = np.tile(np.asarray(x0, dtype=float), (self.iterations, 1))

        return x0, s0

    def _runIterations(self, x0, s0):
        '''
        Run all Monte Carlo simulations from the initial states x0.

        :param x0: Initial continuous state per run, shape [iterations, n].
        :param s0: State ID of the initial region (nr_states if x0 is outside the partition).
        :return:
            - outcome: Why each run ended (index into OUTCOMES).
            - length: Step L at which each run ended.
            - x: States per run and step, shape [iterations, sim_horizon + 1, n]; entries after step L are unused.
            - u: Inputs per run and step, shape [iterations, sim_horizon, p]; entries from step L are unused.
        '''

        tables = _SimTables(
            boundary_lb=jnp.asarray(self.partition.boundary_lb, dtype=jnp.float32),
            boundary_ub=jnp.asarray(self.partition.boundary_ub, dtype=jnp.float32),
            number_per_dim=jnp.asarray(self.partition.number_per_dim, dtype=jnp.int32),
            region_linear_idx=self.partition.region_linear_idx,
            region_linear_state=self.partition.region_linear_state,
            region_linear_strides=self.partition.region_linear_strides,
            is_goal=jnp.asarray(self.is_goal),
            is_critical=jnp.asarray(self.is_critical),
            policy=jnp.asarray(self.policy),
            policy_inputs=jnp.asarray(self.policy_inputs, dtype=jnp.float32),
        )

        outcome, length, x, u = _simulate(self.model.step, self.horizon, tables, jnp.asarray(x0, dtype=jnp.float32),
                                          jnp.int32(s0), jnp.asarray(self.noise, dtype=jnp.float32))

        return np.asarray(outcome), np.asarray(length), np.asarray(x), np.asarray(u)

    def log_outcomes(self, outcome, length, x):
        '''
        Log how many runs ended for each reason, and (if verbose, or at debug level) how each run ended.
        '''

        level = logging.INFO if self.verbose else logging.DEBUG
        if logger.isEnabledFor(level):
            for m, (o, L) in enumerate(zip(outcome.tolist(), length.tolist())):
                logger.log(level, f'- Run {m}: {OUTCOMES[o]} at k = {L} (x_k = {x[m, L]})')

        counts = np.bincount(outcome, minlength=len(OUTCOMES))
        logger.info('Monte Carlo runs: ' + ', '.join(f'{c} {name}' for name, c in zip(OUTCOMES, counts) if c > 0))


class _SimTables(NamedTuple):
    '''Arrays the jitted simulation loop reads (passed as arguments, so they are not compiled in as constants).'''
    boundary_lb: jax.Array
    boundary_ub: jax.Array
    number_per_dim: jax.Array
    region_linear_idx: jax.Array
    region_linear_state: jax.Array
    region_linear_strides: jax.Array
    is_goal: jax.Array
    is_critical: jax.Array
    policy: jax.Array
    policy_inputs: jax.Array


def _x2state(x, tables):
    '''
    Batched version of partition.x2state: the state ID for each point in x [M, n]. Missing states (outside the
    partition boundary, or a cell absent from a sparse partition) get the ID nr_states.

    Float32 division on the CPU backend of XLA is not always correctly rounded, so a point exactly on a cell boundary
    may be assigned to the neighboring cell from the one partition.x2state picks; both cells contain the point.
    '''
    num_states = tables.is_goal.shape[0] - 1
    lb, ub, keys = tables.boundary_lb, tables.boundary_ub, tables.region_linear_idx

    in_boundary = jnp.all((x >= lb) & (x <= ub), axis=1)
    grid_idx = jnp.floor((x - lb) / (ub - lb) * tables.number_per_dim).astype(tables.region_linear_strides.dtype)
    # A point on the upper boundary has grid index number_per_dim, which is not a cell
    in_grid = jnp.all((grid_idx >= 0) & (grid_idx < tables.number_per_dim), axis=1)

    key = jnp.sum(grid_idx * tables.region_linear_strides, axis=1)
    pos = jnp.minimum(jnp.searchsorted(keys, key), keys.shape[0] - 1)
    exists = in_boundary & in_grid & (keys[pos] == key)

    return jnp.where(exists, tables.region_linear_state[pos], num_states)


@partial(jax.jit, static_argnums=(0, 1))
def _simulate(step, horizon, tables, x0, s0, noise):
    '''
    Simulate all runs, each until it ends, with the same checks per step as a single run would make.

    :param step: The model's step function for one state, step(x, u, w).
    :param horizon: Number of simulation steps.
    :param tables: The partition, goal, critical and policy arrays (_SimTables).
    :param x0: Initial state per run, shape [M, n].
    :param s0: State ID of the initial region (all runs start in it).
    :param noise: Noise per run and step, shape [M, horizon, n].
    :return: outcome [M], length [M], x [M, horizon + 1, n], u [M, horizon, p] (see MonteCarloSim._runIterations).
    '''
    step = jax.vmap(step)
    num_states = tables.is_goal.shape[0] - 1
    stationary = tables.policy.ndim == 1
    M = x0.shape[0]

    def cond(carry):
        k, _, outcome, *_ = carry
        return (k <= horizon) & jnp.any(outcome < 0)

    def body(carry):
        k, x, outcome, length, xs, us = carry

        # Determine to which region the state belongs
        s = jnp.where(k == 0, s0, _x2state(x, tables))
        exists = s < num_states
        s_valid = jnp.minimum(s, num_states - 1)

        # Retrieve the action from the policy (with a time index for a finite-horizon policy)
        if stationary:
            action, u = tables.policy[s_valid], tables.policy_inputs[s_valid]
        else:
            t = jnp.minimum(k, tables.policy.shape[0] - 1)
            action, u = tables.policy[t, s_valid], tables.policy_inputs[t, s_valid]

        # The first check that holds decides why a run ends; -1 means it continues
        reason = jnp.select(
            [~exists, tables.is_goal[s], tables.is_critical[s], k >= horizon, action < 0],
            [ABSORBING, GOAL, CRITICAL, HORIZON, NO_ACTION], default=-1)
        ends = (outcome < 0) & (reason >= 0)
        outcome = jnp.where(ends, reason, outcome)
        length = jnp.where(ends, k, length)

        # Advance the unfinished runs; finished runs keep their last state
        w = jax.lax.dynamic_index_in_dim(noise, jnp.minimum(k, horizon - 1), axis=1, keepdims=False)
        x = jnp.where((outcome < 0)[:, None], step(x, u, w), x)

        xs = xs.at[:, k + 1].set(x, mode='drop')
        us = us.at[:, k].set(u, mode='drop')

        return k + 1, x, outcome, length, xs, us

    xs = jnp.zeros((M, horizon + 1, x0.shape[1]), dtype=x0.dtype).at[:, 0].set(x0)
    us = jnp.zeros((M, horizon, tables.policy_inputs.shape[-1]), dtype=tables.policy_inputs.dtype)
    init = (0, x0, jnp.full(M, -1, dtype=jnp.int32), jnp.zeros(M, dtype=jnp.int32), xs, us)

    _, _, outcome, length, xs, us = jax.lax.while_loop(cond, body, init)
    return outcome, length, xs, us
