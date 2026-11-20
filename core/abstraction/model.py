import logging
import time

import jax.numpy as jnp


logger = logging.getLogger(__name__)


def parse_nonlinear_model(model):
    '''
    Parse nonlinear dynamical model

    :param base_model: Input model.
    :return: Model object
    '''

    logger.info('Parse nonlinear dynamical model...')
    t = time.time()

    model.partition['boundary'] = jnp.array(model.partition['boundary']).astype(float)
    model.partition['number_per_dim'] = jnp.array(model.partition['number_per_dim']).astype(int)

    # Control limitations
    model.uMin = jnp.array(model.uMin, dtype=float)
    model.uMax = jnp.array(model.uMax, dtype=float)

    logger.info('Model parsing done (took %.3f sec.)', (time.time() - t))
    return model
