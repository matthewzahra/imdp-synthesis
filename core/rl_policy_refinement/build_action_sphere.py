from core.box import Box
from core.rl_policy_refinement.action_sets import DiscreteFixedSphere
import jax.numpy as jnp
import json

'''
Helper functions to set up the construction of the Sphere class - found in generate_action_sets.py
'''

# find the limits of the action space so that we can clip agent actions to them
def generate_vals_to_clip(model):
	input_space = model.input_space

	if not isinstance(input_space, Box):
		raise ValueError('Input space is not of type Box.')

	return [input_space.lower, input_space.upper]

def generate_action_set(model, continuous=False, dynamic=False):
	if continuous:
		cont = "CONTINUOUS"
	else:
		cont = "DISCRETE"

	if dynamic:
		dyn = "DYNAMIC"
	else:
		dyn = "FIXED"
		
	if continuous and dynamic:
		raise NotImplemented

	elif continuous and not dynamic:
		raise NotImplemented

	elif not continuous and dynamic:
		raise NotImplemented

	else:
		with open("core/rl_policy_refinement/config/fixed_sphere.json", "r") as f:
			data = json.load(f)
			return DiscreteFixedSphere(vals_to_clip=generate_vals_to_clip(model), model=model, radii=jnp.array(data[str(model)][cont][dyn]))