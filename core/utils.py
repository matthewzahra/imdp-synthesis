import logging
from pathlib import Path

import numpy as np


class _CleanConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        if record.levelno >= logging.ERROR:
            return f'ERROR: {message}'
        if record.levelno >= logging.WARNING:
            return f'WARN: {message}'
        if record.levelno == logging.DEBUG:
            return f'DEBUG: {message}'
        return message


def configure_logging(log_level: str) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(_CleanConsoleFormatter())

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, log_level.upper()))
    logging.getLogger('jax._src.xla_bridge').setLevel(logging.WARNING)
    logging.getLogger('fontTools').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)


def add_file_handler(log_dir, stamp: str = '') -> None:
    """Add a file handler that writes all log output to run_<stamp>.log in log_dir."""
    filename = f'run_{stamp}.log' if stamp else 'run.log'
    file_handler = logging.FileHandler(log_dir / filename, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(
        fmt='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%H:%M:%S',
    ))
    logging.getLogger().addHandler(file_handler)


def create_batches(data_length, batch_size):
    '''
    Create batches for the given data and batch size. Returns the start and end indices to iterate over.

    :param data_length: Total number of data points.
    :param batch_size: Number of points per batch.
    :return: Each batch is represented by the slice [starts[i]:ends[i]].
    '''

    num_batches = np.ceil(data_length / batch_size).astype(int)
    starts = (np.arange(num_batches) * batch_size).astype(int)
    ends = (np.minimum(starts + batch_size, data_length)).astype(int)

    return starts, ends


def policy_to_inputs(partition, policy, dtype=np.float32):
    '''
    Control inputs of a policy over the abstract states.

    :param partition: Partition whose regions['actions'] holds the discrete actions, shape
        (1, num_actions, input_dim) for a dense partition or (num_states, num_actions, input_dim) for a sparse one
    :param policy: Action ID per state, with the absorbing state last and -1 where no action is chosen; shape
        [nr_states] (stationary) or [N, nr_states] (one row per time step of a finite horizon)
    :param dtype: Float type of the inputs
    :return: Inputs of shape policy.shape[:-1] + (num_states, input_dim), excluding the absorbing state; NaN
        where no action is chosen
    '''
    policy = np.asarray(policy)
    actions = np.asarray(partition.regions['actions'])
    num_states = policy.shape[-1] - 1

    # A dense partition stores (1, num_actions, input_dim); broadcast to all states.
    if actions.shape[0] == 1:
        actions = np.broadcast_to(actions, (num_states, *actions.shape[1:]))

    state_policy = policy[..., :num_states]
    inputs = np.full(state_policy.shape + (actions.shape[2],), fill_value=np.nan, dtype=dtype)
    mask = state_policy >= 0
    states = np.broadcast_to(np.arange(num_states), state_policy.shape)
    inputs[mask] = actions[states[mask], state_policy[mask]]

    return inputs


def save_values_per_step(output_dir, values_s0):
    '''
    Write the value in the initial state per number of remaining steps, V_k(s0) for k = 0, ..., N, to
    values_per_step.csv (finite horizon only).
    '''
    import csv

    path = Path(output_dir) / 'values_per_step.csv'
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['steps_remaining', 'value_s0'])
        writer.writerows((k, float(v)) for k, v in enumerate(values_s0))
    logging.getLogger(__name__).info('Values per step saved to %s', path)


def lexsort4d(array):
    idxs = np.lexsort((
        array[:, 3],
        array[:, 2],
        array[:, 1],
        array[:, 0]
    ))

    return array[idxs]


def cm2inch(*tupl):
    '''
    Convert centimeters to inches
    '''

    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)


def remove_consecutive_duplicates(trace):
    '''
    Remove consecutive duplicates from a given trace.

    :param trace:
    :return: Trace without duplicates
    '''
    done = False
    i = 0
    while not done:
        # If same as next entry, remove it
        if i >= len(trace) - 1:
            done = True
        else:
            if np.all(trace[i] == trace[i + 1]):
                trace = np.delete(trace, i + 1, axis=0)
            else:
                i += 1

    return trace


def jit_compile_count(jitted_function) -> int | None:
    """Return JIT cache size if available, else ``None``."""
    cache_size_fn = getattr(jitted_function, "_cache_size", None)
    if cache_size_fn is None:
        return None
    try:
        return int(cache_size_fn())
    except Exception:
        return None