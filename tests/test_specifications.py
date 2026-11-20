'''
Tests for finite-horizon reach-avoid and safety specifications on the SVMDP and IMDP paths.

The finite-horizon solvers are checked against plain reference implementations of the Bellman step, which
enumerate every successor box and cell (SVMDP) or solve a linear program per state-action pair (IMDP), on a
small 2D model with a goal and an obstacle. Run from the repository root with: python -m pytest tests
'''

import itertools

import numpy as np
import pytest
from scipy.optimize import linprog

from benchmarks.affine import AffineDynamics
from core.abstraction.partition import DensePartition
from core.box import Box
from core.jax_config import configure_jax
from core.options import parse_arguments
from core.specification import (REACH_AVOID, SAFETY, check_solver_support, initial_values,
                                 resolve_specification)
from core.utils import policy_to_inputs
from core.validate.simulate import MonteCarloSim

ATOL = 1e-5  # The solvers work in float32


class SmallModel(AffineDynamics):
    '''Small 2D model with a goal (reach-avoid only) and an obstacle, coarse enough for the references.'''

    def __init__(self, args, spec, horizon=None):
        AffineDynamics.__init__(self, args,
                                A=[[1, 0.5], [0, 0.9]], B=[[0.125], [0.5]], q=[0, 0],
                                gaussian=dict(variance=[0.05, 0.05], cells=[15, 15]))
        self.plot_dimensions = [0, 1]
        self.state_space = Box([-4.0, -2.0], [4.0, 2.0], names=self.state_variables)
        self.state_grid = [8, 6]
        self.input_space, self.input_grid = Box([-1.0], [1.0]), [3]
        self.goal = [Box([2.0, -2.0], [4.0, 2.0])] if spec == REACH_AVOID else []
        self.critical = [Box([-1.0, -2.0], [0.0, -0.5])]
        self.x0 = np.array([-2.5, 0.0])
        self.spec = spec
        self.horizon = horizon


def make_args(*argv):
    args = parse_arguments(['--model', 'none', '--dense', *argv])
    configure_jax(args)
    return args


def build_model(spec, horizon, *argv):
    args = make_args(*argv)
    model = SmallModel(args, spec, horizon)
    model.check()
    resolve_specification(model, args)
    return args, model


def cell_value(V, partition, grid_idx, wrap):
    '''Value of a grid cell: 0 outside the partition (the absorbing state).'''
    idx = []
    for d, c in enumerate(grid_idx):
        n = int(partition.number_per_dim[d])
        if wrap[d]:
            c = c % n
        elif not 0 <= c < n:
            return 0.0
        idx.append(c)
    state, exists = partition.grid_idx2state(idx)
    return float(V[state]) if exists else 0.0


def check_monte_carlo(model, partition, values, policy, fixed, horizon, spec, n=2000):
    '''
    Simulate the policy from the center of the non-fixed state whose bound is closest to 0.5 (so that the check is
    informative), and check that the empirical rate is not significantly below the bound.
    '''
    candidates = np.flatnonzero(~fixed)
    s = candidates[np.argmin(np.abs(values[candidates] - 0.5))]
    bound = float(values[s])
    assert 0 < bound < 1
    x = (np.asarray(partition.regions['lower_bounds'][s]) + np.asarray(partition.regions['upper_bounds'][s])) / 2
    np.random.seed(0)
    sim = MonteCarloSim(model, partition, policy, policy_to_inputs(partition, policy), x,
                        iterations=n, sim_horizon=horizon, verbose=False, spec=spec)
    assert sim.results['satprob'] >= bound - 3 * np.sqrt(bound * (1 - bound) / n) - 0.01


# ---------------------------------------------------------------------------------------------------------
# SVMDP


def build_svmdp(spec, horizon):
    from core.abstraction.svmdp.forward_reachability import RectangularForward
    from core.abstraction.svmdp.successor_ids import TiledGrid
    from core.abstraction.svmdp.svmdp import SVMDP

    args, model = build_model(spec, horizon)
    partition = DensePartition(model=model)
    frs = RectangularForward(args=args, partition=partition, model=model)
    svmdp = SVMDP(partition=partition, states=np.array(partition.regions['idxs']), x0=model.x0,
                  goal_regions=np.array(partition.goal['bools']),
                  critical_regions=np.array(partition.critical['bools']),
                  interval_lb=frs.interval_lb, interval_ub=frs.interval_ub, box_probs=frs.box_probs,
                  slots=frs.slots, max_slice=frs.max_slice, union_span=frs.union_span,
                  grid=TiledGrid(partition=partition, wrap=model.wrap), A_id=list(range(frs.num_actions)),
                  P_absorbing=model.noise.partition['remainder'])
    # The DP releases the successor arrays; keep copies to restore them between solver calls
    arrays = (frs.interval_lb.copy(), frs.interval_ub.copy(), frs.box_probs.copy())
    return args, model, partition, svmdp, arrays


def svmdp_reference_step(V, partition, wrap, arrays, slots, fixed):
    '''One robust Bellman step by enumerating every successor box and every cell in it.'''
    lb, ub, probs = arrays
    bounds = np.cumsum((0,) + tuple(slots))
    V_new = V.copy()
    for s in np.flatnonzero(~fixed):
        best = 0.0
        for a in range(lb.shape[1]):
            total = 0.0
            for box, combo in enumerate(itertools.product(*[range(k) for k in slots])):
                p = probs[s, a, box]
                if p == 0:
                    continue
                ranges = [range(lb[s, a, bounds[d] + combo[d]], ub[s, a, bounds[d] + combo[d]] + 1)
                          for d in range(len(slots))]
                total += p * min(cell_value(V, partition, c, wrap) for c in itertools.product(*ranges))
            best = max(best, min(max(total, 0.0), 1.0))
        V_new[s] = best
    return V_new


def run_svmdp_finite(args, svmdp, arrays, s0, horizon, spec):
    from core.abstraction.svmdp.dynprog import SVMDP_DP_finite
    svmdp.interval_lb, svmdp.interval_ub, svmdp.box_probs = (a.copy() for a in arrays)
    return SVMDP_DP_finite(args=args, svmdp=svmdp, s0=s0, horizon=horizon, spec=spec)


@pytest.fixture(scope='module', params=[REACH_AVOID, SAFETY])
def svmdp_case(request):
    spec = request.param
    return (spec,) + build_svmdp(spec, horizon=5)


def test_svmdp_finite_matches_reference(svmdp_case):
    spec, args, model, partition, svmdp, arrays = svmdp_case
    s0 = partition.x2state(model.x0)[0]
    V0, fixed = initial_values(svmdp.goal_regions, svmdp.critical_regions, spec)

    V_ref = [V0]
    for _ in range(15):
        V_ref.append(svmdp_reference_step(V_ref[-1], partition, model.wrap, arrays, svmdp.slots, fixed))

    for horizon in (1, 2, 5, 15):
        values, policy, values_s0 = run_svmdp_finite(args, svmdp, arrays, s0, horizon, spec)
        np.testing.assert_allclose(values[:-1], V_ref[horizon], atol=ATOL)
        np.testing.assert_allclose(values_s0, [V[s0] for V in V_ref[:horizon + 1]], atol=ATOL)
        assert values[-1] == 0  # absorbing state
        assert policy.shape == (horizon, svmdp.nr_states)
        assert np.all(policy[:, :-1][:, fixed] == -1) and np.all(policy[:, :-1][:, ~fixed] >= 0)


def test_svmdp_finite_monotone_and_converges(svmdp_case):
    from core.abstraction.svmdp.dynprog import SVMDP_DP

    spec, args, model, partition, svmdp, arrays = svmdp_case
    s0 = partition.x2state(model.x0)[0]
    V_by_horizon = {N: run_svmdp_finite(args, svmdp, arrays, s0, N, spec)[0] for N in (1, 3, 10, 400)}

    if spec == REACH_AVOID:
        # More steps can only help to reach the goal, up to the unbounded-horizon value
        assert np.all(np.diff(np.stack([V_by_horizon[N] for N in (1, 3, 10, 400)]), axis=0) >= -ATOL)
        svmdp.interval_lb, svmdp.interval_ub, svmdp.box_probs = (a.copy() for a in arrays)
        V_inf, _ = SVMDP_DP(args=args, svmdp=svmdp, s0=s0, max_iterations=10000, epsilon=1e-7)
        assert np.all(V_by_horizon[10] <= V_inf + 1e-4)
        np.testing.assert_allclose(V_by_horizon[400], V_inf, atol=1e-3)
    else:
        # Staying safe for more steps can only be harder
        assert np.all(np.diff(np.stack([V_by_horizon[N] for N in (1, 3, 10, 400)]), axis=0) <= ATOL)


def test_svmdp_monte_carlo_respects_bound(svmdp_case):
    spec, args, model, partition, svmdp, arrays = svmdp_case
    s0 = partition.x2state(model.x0)[0]
    horizon = 15 if spec == REACH_AVOID else 5
    values, policy, _ = run_svmdp_finite(args, svmdp, arrays, s0, horizon, spec)
    _, fixed = initial_values(svmdp.goal_regions, svmdp.critical_regions, spec)
    check_monte_carlo(model, partition, values, policy, fixed, horizon, spec)


# ---------------------------------------------------------------------------------------------------------
# IMDP


def build_imdp(spec, horizon):
    from core.abstraction.imdp.forward_reachability import RectangularForward
    from core.abstraction.imdp.imdp import IMDP
    from core.abstraction.imdp.probability_intervals import compute_probability_intervals

    args, model = build_model(spec, horizon)
    partition = DensePartition(model=model)
    actions = RectangularForward(args=args, partition=partition, model=model)
    P_full, S_id, A_id, P_absorbing = compute_probability_intervals(args=args, model=model, partition=partition,
                                                                     actions=actions, vectorized=True)
    imdp = IMDP(partition=partition, states=np.array(partition.regions['idxs']), x0=model.x0,
                goal_regions=np.array(partition.goal['bools']),
                critical_regions=np.array(partition.critical['bools']),
                P_full=P_full, S_id=S_id, A_id=A_id, P_absorbing=P_absorbing)
    return args, model, partition, imdp


def imdp_reference_step(V, imdp, fixed):
    '''One robust Bellman step, solving the inner minimization over the intervals as a linear program.'''
    V_new = V.copy()
    for s in np.flatnonzero(~fixed):
        if s not in imdp.A_id or len(imdp.A_id[s]) == 0:
            V_new[s] = 0.0  # No enabled action: fails once a step remains
            continue
        best = 0.0
        for a in range(len(imdp.A_id[s])):
            lbs = np.append(imdp.P_full[s][a, :, 0], imdp.P_absorbing[s][a, 0])
            ubs = np.append(imdp.P_full[s][a, :, 1], imdp.P_absorbing[s][a, 1])
            values = np.append(V[imdp.S_id[s][a]], 0.0)
            res = linprog(values, A_eq=np.ones((1, len(values))), b_eq=[1.0],
                          bounds=list(zip(lbs, ubs)), method='highs')
            assert res.status == 0, f'Infeasible probability intervals in state {s}, action {a}'
            best = max(best, min(max(res.fun, 0.0), 1.0))
        V_new[s] = best
    return V_new


@pytest.fixture(scope='module', params=[REACH_AVOID, SAFETY])
def imdp_case(request):
    spec = request.param
    return (spec,) + build_imdp(spec, horizon=5)


def test_imdp_finite_matches_reference(imdp_case):
    from core.abstraction.imdp.rvi_jax import RVI_JAX_finite

    spec, args, model, partition, imdp = imdp_case
    s0 = partition.x2state(model.x0)[0]
    V0, fixed = initial_values(imdp.goal_regions, imdp.critical_regions, spec)

    V_ref = [np.append(V0, 0.0)]
    for _ in range(15):
        V_ref.append(imdp_reference_step(V_ref[-1], imdp, np.append(fixed, True)))

    for horizon in (1, 2, 5, 15):
        values, policy, values_s0 = RVI_JAX_finite(args=args, imdp=imdp, s0=s0, horizon=horizon, spec=spec)
        np.testing.assert_allclose(values, V_ref[horizon], atol=ATOL)
        np.testing.assert_allclose(values_s0, [V[s0] for V in V_ref[:horizon + 1]], atol=ATOL)
        assert policy.shape == (horizon, imdp.nr_states)


def test_imdp_finite_monotone_and_converges(imdp_case):
    from core.abstraction.imdp.rvi_jax import RVI_JAX, RVI_JAX_finite

    spec, args, model, partition, imdp = imdp_case
    s0 = partition.x2state(model.x0)[0]
    V = {N: RVI_JAX_finite(args=args, imdp=imdp, s0=s0, horizon=N, spec=spec)[0] for N in (1, 3, 10, 400)}
    stacked = np.stack([V[N] for N in (1, 3, 10, 400)])
    if spec == REACH_AVOID:
        assert np.all(np.diff(stacked, axis=0) >= -ATOL)
        V_inf, _ = RVI_JAX(args=args, imdp=imdp, s0=s0, max_iterations=10000, epsilon=1e-7)
        np.testing.assert_allclose(V[400], V_inf, atol=1e-3)
    else:
        assert np.all(np.diff(stacked, axis=0) <= ATOL)


def test_imdp_monte_carlo_respects_bound(imdp_case):
    from core.abstraction.imdp.rvi_jax import RVI_JAX_finite

    spec, args, model, partition, imdp = imdp_case
    s0 = partition.x2state(model.x0)[0]
    horizon = 15 if spec == REACH_AVOID else 5
    values, policy, _ = RVI_JAX_finite(args=args, imdp=imdp, s0=s0, horizon=horizon, spec=spec)
    _, fixed = initial_values(imdp.goal_regions, imdp.critical_regions, spec)
    check_monte_carlo(model, partition, values, policy, fixed, horizon, spec)


# ---------------------------------------------------------------------------------------------------------
# Specification handling and helpers


def test_safety_requires_finite_horizon():
    with pytest.raises(ValueError, match='finite horizon'):
        build_model(SAFETY, None)


def test_horizon_option_overrides_benchmark():
    _, model = build_model(REACH_AVOID, 7, '--horizon', '3')
    assert model.horizon == 3


def test_safety_rejects_goal_and_sparse_mode():
    args = make_args('--horizon', '4')
    model = SmallModel(args, SAFETY, 4)
    model.goal = [Box([2.0, -2.0], [4.0, 2.0])]
    with pytest.raises(ValueError, match='no goal set'):
        resolve_specification(model, args)

    args = parse_arguments(['--model', 'none', '--horizon', '4'])
    model = SmallModel(args, SAFETY, 4)
    with pytest.raises(ValueError, match='--dense'):
        resolve_specification(model, args)


def test_storm_rejects_finite_horizon_and_safety():
    for spec, horizon in ((REACH_AVOID, 5), (SAFETY, 5)):
        args, model = build_model(spec, horizon, '--solver', 'storm')
        with pytest.raises(NotImplementedError):
            check_solver_support(model, args)
    args, model = build_model(REACH_AVOID, None, '--solver', 'storm')
    check_solver_support(model, args)  # Unbounded reach-avoid stays supported


def test_initial_values():
    goal = np.array([True, False, False, False])
    critical = np.array([False, True, False, True])
    V0, fixed = initial_values(goal, critical, REACH_AVOID)
    np.testing.assert_array_equal(V0, [1, 0, 0, 0])
    np.testing.assert_array_equal(fixed, [True, True, False, True])
    V0, fixed = initial_values(np.zeros(4, dtype=bool), critical, SAFETY)
    np.testing.assert_array_equal(V0, [1, 0, 1, 0])
    np.testing.assert_array_equal(fixed, critical)


def test_policy_to_inputs_stationary_and_time_dependent():
    class Partition:
        regions = {'actions': np.array([[[-1.0], [0.0], [1.0]]])}  # Dense: shared by all states

    policy = np.array([2, -1, 0, -1])  # 3 states and the absorbing state
    inputs = policy_to_inputs(Partition, policy)
    assert inputs.shape == (3, 1)
    np.testing.assert_array_equal(inputs[[0, 2], 0], [1.0, -1.0])
    assert np.isnan(inputs[1, 0])

    policy_t = np.stack([policy, np.array([0, 1, 2, -1])])
    inputs_t = policy_to_inputs(Partition, policy_t)
    assert inputs_t.shape == (2, 3, 1)
    np.testing.assert_array_equal(inputs_t[1, :, 0], [-1.0, 0.0, 1.0])
