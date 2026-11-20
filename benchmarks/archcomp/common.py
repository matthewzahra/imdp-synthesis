import jax.numpy as jnp
import numpy as np

from core.box import Box


def stack(*values):
    '''
    Stack per-dimension bounds into a vector. The interval functions of benchmarks.dynamics.setmath return scalars or
    arrays of shape [1] (mult), so they are flattened first.
    '''
    return jnp.concatenate([jnp.ravel(v) for v in values])


def no_control(model, n):
    '''
    Give an autonomous system (a verification benchmark) a single dummy input that does not affect the dynamics.

    :return: Input matrix B = 0 of shape [n, 1], for AffineDynamics
    '''
    model.input_space, model.input_grid = Box([0.0], [0.0]), [1]
    return np.zeros((n, 1))
