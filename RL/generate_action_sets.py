import numpy as np
import jax.numpy as jnp
import jax
from RL.helper_functions import distance_from_box_to_box

'''
Here we write functions ot help us compute F(x,a), where x is a concrete state and a an abstract action (or similar?)
'''

class Spheres:
	'''
	class to dynamically generate action spheres.
	they may be dependent on different dimensions, clipping and wrap around, as well as proximity to critical/goal regions
	'''
	def __init__(
			self,
			thresholds,
			radii_options,
			vals_to_clip,
			vals_to_wrap,
			critical_regions
		):
		self.thresholds = thresholds
		self.radii_options = radii_options
		self.vals_to_clip = vals_to_clip
		self.vals_to_wrap = vals_to_wrap
		self.critical_regions = critical_regions

	# the most basic one is simply an L_infinity ball around a with a given direction - this forms a hyperrectangle
	# we must clip it so that if we are on the extreme of the action space, we don't allow invalid actions (if lower and upper bounds provided)
	def L_infinity(self, centre, distances, lower_bounds=None, upper_bounds=None):
		'''
		:param centre: centre coordinates of the ball
		:param distance: radius for each dimension - it does not need to be the same everywhere. Assumes all values are non-negative!!!

		return: lower bounds, upper bounds
		'''
		assert centre.size == distances.size
		if lower_bounds is not None or upper_bounds is not None:
			return jnp.maximum(centre-distances,lower_bounds),jnp.minimum(centre+distances, upper_bounds) # TODO - check this return type - should it be in a list? 
		else:
			return centre-distances, centre+distances


	def wrap_interval(self, x, lower, upper):
		width = upper - lower
		return (x - lower) % width + lower


	def separate_lower_and_upper_bounds_to_jnp_arrays(self,bounds):
		lower = jnp.array([v[0] if v is not None else 0 for v in bounds])
		upper = jnp.array([v[1] if v is not None else 0 for v in bounds])
		return lower,upper


	# function that generates irregular spheres - i.e. we allow for different radii in each dimension
	# we also allow for some values ot be clipped and some values to be wrapped
	def generate_sphere(self,centre, radii, vals_to_clip, vals_to_wrap):
		'''
		given a centre, radii for each dimension and lower/upper bounds for dimension=wise clippping/wrapping, produce the (irregular) sphere
		'''
		lb,ub = self.L_infinity(centre, radii)

		# deal with clipping
		clip_mask = jnp.array([v is not None for v in vals_to_clip])

		clip_lower,clip_upper = self.separate_lower_and_upper_bounds_to_jnp_arrays(vals_to_clip)

		lb_clipped = jnp.clip(lb,clip_lower, clip_upper)
		ub_clipped = jnp.clip(ub,clip_lower, clip_upper)

		# only apply clipping to the right places
		lb = jnp.where(clip_mask, lb_clipped, lb)
		ub = jnp.where(clip_mask, ub_clipped, ub)

		# TODO - deal with lower and upper bounds swapping after wrapping
		# deal with wrapping
		wrap_mask = jnp.array([v is not None for v in vals_to_wrap])
		wrap_lower,wrap_upper = self.separate_lower_and_upper_bounds_to_jnp_arrays(vals_to_wrap)
		
		lb_wrapped = self.wrap_interval(lb,wrap_lower,wrap_upper)
		ub_wrapped = self.wrap_interval(ub,wrap_lower,wrap_upper)

		lb = jnp.where(wrap_mask, lb_wrapped, lb)
		ub = jnp.where(wrap_mask, ub_wrapped, ub)
		return lb,ub

	def get_action_sphere(self, action_centre, state=None, state_min=None,state_max=None):
		'''
		for a given action, give the sphere - this is dependent on the current state.
		use state_min/max when calculating forward reachable sets, so calculating the action spheres for a set (box) of states

		This is the "public" facing function that we expect to get called elsewhere
		'''

		# make the box a single point if we are just working with 1 single state
		if state_min is None or state_max is None:
			state_min = state
			state_max = state

		# find the distance between input and the nearest critical region
		calc_distances = jax.vmap(
            lambda lb, ub: distance_from_box_to_box(state_min,state_max,lb,ub),
        )

		critical_distances = calc_distances(self.critical_regions[:,0,:], self.critical_regions[:,1,:])
		closest_critical = jnp.min(critical_distances)

		mask = closest_critical >= self.thresholds
		radii = jnp.where(
			mask[:, None],
			self.radii_options,
			0
		)

		# pick the largest radii values that are allowed
		idx = len(mask) - jnp.sum(mask)
		idx = jnp.maximum(idx,0)

		return self.generate_sphere(action_centre,radii[idx],self.vals_to_clip,self.vals_to_wrap)

