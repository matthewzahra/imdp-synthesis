'''
Specifications: reach-avoid (the default) and safety, each over an unbounded or a finite horizon.

A benchmark may set two attributes; both are optional:
- `spec`: 'reach_avoid' (default) or 'safety'
- `horizon`: number of steps N (default None, i.e., unbounded). `--horizon N` overrides it.

A horizon N counts transitions, so the states x_0, ..., x_N are all checked. The absorbing state (outside
the partition, or noise beyond the truncated noise support) always counts as a failure.
- reach_avoid, unbounded: reach a goal cell at some time, without visiting a critical cell before.
- reach_avoid, horizon N: the same, but the goal must be reached within N steps.
- safety, horizon N: avoid the critical cells (and the absorbing state) at all times 0, ..., N. The state
  space boundary therefore acts as the safe set.

Safety over an unbounded horizon is not supported. With unbounded noise, the truncated noise tail counts
as a failure at every step, so the bound tends to zero. Moreover, value iteration for safety converges
from above, so its iterates are not lower bounds.
'''

import logging

import numpy as np

logger = logging.getLogger(__name__)

REACH_AVOID = 'reach_avoid'
SAFETY = 'safety'
SPEC_TYPES = (REACH_AVOID, SAFETY)


def get_spec(model):
    '''
    Specification type and horizon of a model.

    Models from older checkpoints do not have these attributes; they are unbounded reach-avoid.

    :param model: Model object
    :return: Tuple (spec, horizon), where horizon is None for an unbounded horizon
    '''
    return getattr(model, 'spec', REACH_AVOID), getattr(model, 'horizon', None)


def describe_spec(model):
    '''Short human-readable description of the specification, e.g., "safety (horizon 6)".'''
    spec, horizon = get_spec(model)
    return f"{spec} ({'unbounded horizon' if horizon is None else f'horizon {horizon}'})"


def resolve_specification(model, args):
    '''
    Set model.spec and model.horizon from the benchmark and the command line, and validate them.

    :param model: Benchmark model (before parsing)
    :param args: Argument namespace (provides horizon, dense, satprob)
    '''
    spec = getattr(model, 'spec', REACH_AVOID)
    horizon = getattr(model, 'horizon', None)
    if getattr(args, 'horizon', None) is not None:
        horizon = args.horizon

    if spec not in SPEC_TYPES:
        raise ValueError(f"Unknown specification '{spec}'. Expected one of {SPEC_TYPES}.")

    if horizon is not None:
        horizon = int(horizon)
        if horizon < 1:
            raise ValueError(f'The horizon must be a positive number of steps, got {horizon}.')

    if spec == SAFETY:
        if horizon is None:
            raise ValueError(
                'Safety specifications need a finite horizon: set `horizon` in the benchmark or pass --horizon N. '
                '(Over an unbounded horizon, the truncated noise tail drives the bound to zero.)')
        if len(model.goal) > 0:
            raise ValueError('Safety specifications have no goal set; the benchmark defines a nonempty `goal`.')
        if not getattr(args, 'dense', False):
            raise ValueError(
                'Safety specifications need --dense: the RL front-end only supports reach-avoid rewards.')

    if horizon is not None and getattr(args, 'satprob', 1) < 1:
        logger.warning('--satprob is ignored for a finite horizon: all %d steps of the dynamic program must run '
                       'before the value in the initial state is known.', horizon)

    model.spec = spec
    model.horizon = horizon
    logger.info('- Specification: %s', describe_spec(model))


def check_solver_support(model, args):
    '''
    Raise an error for specifications that the selected IMDP solver backend does not support.

    Called before the abstraction is built, so that an unsupported combination fails early.
    '''
    spec, horizon = get_spec(model)
    if getattr(args, 'solver', 'jax') == 'storm' and (spec != REACH_AVOID or horizon is not None):
        raise NotImplementedError(
            f'The Storm backend only supports reach-avoid over an unbounded horizon (got {describe_spec(model)}). '
            'Use --solver jax: a finite horizon needs a policy per time step, which Storm does not return.')


def initial_values(goal_regions, critical_regions, spec):
    '''
    Values with zero steps remaining (V_0) and the states whose value is fixed, over the partition states.

    Goal states (reach-avoid) are fixed at 1 and critical states at 0. All other states are updated by the
    dynamic program; with zero steps remaining, they are worth 1 under safety (they are safe now) and 0
    under reach-avoid (the goal is not reached).

    :param goal_regions: Boolean mask over the partition states
    :param critical_regions: Boolean mask over the partition states
    :param spec: 'reach_avoid' or 'safety'
    :return: Tuple (V0, fixed) of a float array and a boolean mask over the partition states
    '''
    goal = np.asarray(goal_regions, dtype=bool)
    critical = np.asarray(critical_regions, dtype=bool)
    if spec == SAFETY:
        V0 = (~critical).astype(float)
        fixed = critical.copy()
    elif spec == REACH_AVOID:
        V0 = goal.astype(float)
        fixed = goal | critical
    else:
        raise ValueError(f"Unknown specification '{spec}'.")
    return V0, fixed
