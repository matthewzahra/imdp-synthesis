'''
Checks of the ARCH-COMP benchmarks that need no abstraction (so they are fast and use little memory): every benchmark
builds and has a valid specification, its initial state is safe, its sets fall on its grid as documented, and its
interval bounds (step_set) contain the successors of sampled states.
'''

import numpy as np
import pytest

from benchmarks import create_model
from benchmarks.archcomp import ARCHCOMP_BENCHMARKS
from core.abstraction.partition import DensePartition, input_grid_points
from core.jax_config import configure_jax
from core.options import parse_arguments


def build(name):
    args = parse_arguments(['--model', name, '--dense'])
    configure_jax(args)
    return create_model(args)


@pytest.fixture(scope='module', params=sorted(ARCHCOMP_BENCHMARKS))
def benchmark(request):
    model = build(request.param)
    return request.param, model, DensePartition(model=model)


def test_specification_and_initial_state(benchmark):
    name, model, partition = benchmark
    assert model.spec in ('reach_avoid', 'safety')
    if model.spec == 'safety':
        assert model.horizon is not None and len(model.goal) == 0
    else:
        assert int(np.sum(partition.goal['bools'])) > 0, f'{name}: no goal cell'

    s0, exists = partition.x2state(model.x0)
    assert exists, f'{name}: x0 outside the partition'
    assert not bool(partition.critical['bools'][s0]), f'{name}: x0 in a critical cell'
    assert not bool(partition.goal['bools'][s0]), f'{name}: x0 already in the goal'


def test_step_set_contains_successors(benchmark):
    '''The bounds of every sampled cell-sized box and action contain the noise-free successors of its points.'''
    name, model, _ = benchmark
    rng = np.random.default_rng(0)
    lower, upper = model.state_space.lower, model.state_space.upper
    width = (upper - lower) / np.asarray(model.state_grid)
    u_grid = input_grid_points(model)
    for _ in range(40):
        box_lb = lower + rng.random(model.n) * (upper - lower - width)
        box_ub = box_lb + width
        u = np.array([rng.choice(g) for g in u_grid])
        next_lb, next_ub = (np.asarray(v) for v in model.step_set(box_lb, box_ub, u, u))
        points = box_lb + rng.random((50, model.n)) * width
        successors = np.array([np.asarray(model.step(x, u, np.zeros(model.n))) for x in points])
        tol = 1e-5 * (1 + np.abs(successors))
        assert np.all(successors >= next_lb - tol) and np.all(successors <= next_ub + tol), \
            f'{name}: a successor lies outside the bounds of box [{box_lb}, {box_ub}] under input {u}'


@pytest.mark.parametrize('name, num_actions, goal_cells, critical_cells', [
    ('IntegratorChain2D', 5, 32 * 32, 0),
    ('Anaesthesia_reach', 16, 8 * 20 * 20, 0),
    ('PatrolRobot_reachavoid', 441, 4 * 4, 8 * 8),
    ('PatrolRobot_reach', 121, 2 * 2, 0),
    ('AutonomousVehicle3D', 99, 10 * 10 * 34, 11 * 2 * 34),
    ('AutonomousVehicle3D_coarse', 30, 10 * 10 * 17, 11 * 2 * 17),
    ('VanDerPol', 10, 3 * 9, 0),
    ('VanDerPol_SySCoRe', 10, 7 * 9, 0),
])
def test_grids_as_documented(name, num_actions, goal_cells, critical_cells):
    model = build(name)
    partition = DensePartition(model=model)
    assert int(np.prod(model.input_grid)) == num_actions
    assert int(np.sum(partition.goal['bools'])) == goal_cells
    assert int(np.sum(partition.critical['bools'])) == critical_cells
