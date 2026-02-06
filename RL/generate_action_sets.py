import numpy as np

'''
Here we write functions ot help us compute F(x,a), where x is a concrete state and a an abstract action (or similar?)
'''

# TODO - make this more modular so that we can better implement interface functions that can be easily swapped in and out 

# the most basic one is simply an L_infinity ball around a with a given direction - this forms a hyperrectangle
# we must clip it so that if we are on the extreme of the action space, we don't allow invalid actions (if lower and upper bounds provided)
def L_infinity(centre, distances, lower_bounds=None, upper_bounds=None):
	'''
	:param centre: centre coordinates of the ball
	:param distance: radius for each dimension - it does not need to be the same everywhere. Assumes all values are non-negative!!!

	return: lower bounds, upper bounds
	'''

	assert centre.size == distances.size
	if lower_bounds is not None or upper_bounds is not None:
		return np.maximum(centre-distances,lower_bounds),np.minimum(centre+distances, upper_bounds) # TODO - check this return type - should it be in a list? 
	else:
		return centre-distances, centre+distances