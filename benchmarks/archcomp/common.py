import jax.numpy as jnp
import numpy as np


def set_grid(model, boundary, number_per_dim):
    '''
    Set the partition of the state space: a grid of number_per_dim cells over the box boundary = [lower, upper].

    Leaving this box counts as a failure, so for safety specifications it is the safe set.
    '''
    model.partition = {
        'boundary': np.array(boundary, dtype=float),
        'number_per_dim': np.array(number_per_dim, dtype=int),
    }
    model.partition['boundary_jnp'] = jnp.array(model.partition['boundary'])
    model.targets = {}


def stack(*values):
    '''
    Stack per-dimension bounds into a vector. The interval functions of benchmarks.dynamics.setmath return scalars or
    arrays of shape [1] (mult), so they are flattened first.
    '''
    return jnp.concatenate([jnp.ravel(v) for v in values])


def boxes(*bounds):
    '''
    Array of boxes for goal or critical sets, each given as [lower, upper]; an empty array if none.

    The partition takes as goal the cells fully contained in a goal box, and as critical the cells that overlap a
    critical box, so a box that is not aligned with the grid is shrunk (goal) or enlarged (critical). Both are
    conservative.
    '''
    return np.array(bounds, dtype=float)


def no_control(model, n):
    '''
    Give an autonomous system (a verification benchmark) a single dummy input that does not affect the dynamics.

    :return: Input matrix B = 0 of shape [n, 1], for AffineDynamics
    '''
    model.uMin, model.uMax, model.num_actions = [0.0], [0.0], [1]
    return np.zeros((n, 1))
