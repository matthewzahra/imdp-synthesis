# One module per system, holding its dynamics and its benchmarks
from math import prod

import numpy as np

from .affine import DoubleIntegrator, Test1D
from .archcomp import ARCHCOMP_BENCHMARKS
from .cartpole import CartPole, CartPole_hard
from .quadcopter import Quadcopter2D, Quadcopter2D_2agent, Quadcopter2D_battery, Quadcopter3D, Quadcopter3D_battery, Quadcopter3D_small
from .dubins import Dubins3D, Dubins4D
from .mountain_car import MountainCar
from .pendulum import Pendulum

# The benchmarks by name (the --model option)
BENCHMARKS = {cls.__name__: cls for cls in (
    Test1D, DoubleIntegrator, Pendulum, MountainCar, Dubins3D, Dubins4D, CartPole, CartPole_hard,
    Quadcopter2D, Quadcopter2D_battery, Quadcopter2D_2agent, Quadcopter3D, Quadcopter3D_small, Quadcopter3D_battery,
)}
BENCHMARKS.update(ARCHCOMP_BENCHMARKS)


def _fit_grid_budget(default_counts, budget):
	'''
	Scale each grid dimension proportionally and round to the nearest integer.
	'''
	
	if budget <= 0:
		raise ValueError("Partition budgets must be positive.")

	defaults = np.array(default_counts, dtype=int)
	if any(count <= 0 for count in defaults):
		raise ValueError("Partition and action grid sizes must be positive.")
	if budget < 1:
		return [1] * len(defaults)

	scale = (budget / prod(defaults)) ** (1 / len(defaults))
	counts = [max(1, round(count * scale)) for count in defaults]

	return counts


def create_model(args):
	from core.specification import resolve_specification

	model_cls = BENCHMARKS.get(args.model)
	if model_cls is None:
		raise ValueError(f"The passed model '{args.model}' could not be found")

	model = model_cls(args)

	states_budget = getattr(args, 'fix_num_states', None)
	actions_budget = getattr(args, 'fix_num_actions', None)
	choices_budget = getattr(args, 'fix_num_choices', None)

	if actions_budget is not None:
		model.input_grid = _fit_grid_budget(model.input_grid, actions_budget)

	if choices_budget is not None:
		model.state_grid = _fit_grid_budget(model.state_grid, choices_budget / np.prod(model.input_grid))
	elif states_budget is not None:
		model.state_grid = _fit_grid_budget(model.state_grid, states_budget)

	model.check()
	resolve_specification(model, args)

	return model
