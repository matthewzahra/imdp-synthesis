'''
Here we write functions ot help us compute F(x,a), where x is a concrete state and a an abstract action (or similar?)
'''

# TODO - make this more modular so that we can better implement interface functions that can be easily swapped in and out 

# the most basic one is simply an L_infinity ball around a with a given direction - this forms a hyperrectangle
def L_infinity(centre,distances):
	'''
	:param centre: centre coordinates of the ball
	:param distance: radius for each dimension - it does not need to be the same everywhere. Assumes all values are non-negative!!!

	return: lower bounds, upper bounds
	'''

	assert centre.size == distances.size
	return centre-distances,centre+distances # TODO - check this return type - should it be in a list? 