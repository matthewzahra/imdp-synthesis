from core.box import Box

'''
Helper functions to set up the construction of the Sphere class - found in generate_action_sets.py
'''

# find the limits of the action space so that we can clip agent actions to them
def generate_vals_to_clip(model):
	input_space = model.input_space

	if not isinstance(input_space, Box):
		raise ValueError('Input space is not of type Box.')

	return [input_space.lower, input_space.upper]