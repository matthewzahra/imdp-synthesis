# Load the benchmarks from the subfiles
from math import prod

import numpy as np

from .Dubins3D import Dubins3D
from .Dubins4D import Dubins4D
from .Drone4D import Drone4D, Drone4D_battery
from .Drone4D_2agent import Drone4D_2agent
from .Drone6D import Drone6D, Drone6D_small, Drone6D_battery
from .Pendulum import Pendulum
from .MountainCar import MountainCar
from .CartPole import CartPole
from .CartPole_hard import CartPole_hard
from .Integrators import DoubleIntegrator
from .archcomp import ARCHCOMP_BENCHMARKS
from .Test1D import Test1D


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

	model_map = {
		'Dubins3D': Dubins3D,
		'Dubins4D': Dubins4D,
		'Drone4D': Drone4D,
		'Drone4D_battery': Drone4D_battery,
		'Drone4D_2agent': Drone4D_2agent,
		'Drone6D': Drone6D,
		'Drone6D_small': Drone6D_small,
		'Drone6D_battery': Drone6D_battery,
		'Pendulum': Pendulum,
		'MountainCar': MountainCar,
		'CartPole': CartPole,
		'CartPole_hard': CartPole_hard,
		'DoubleIntegrator': DoubleIntegrator,
		'Test1D': Test1D,
		**ARCHCOMP_BENCHMARKS,
	}

	model_cls = model_map.get(args.model)
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
