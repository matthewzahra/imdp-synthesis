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
			vals_to_clip,
			vals_to_wrap,
			critical_regions,
			model,
			continuous=False,
			thresholds=None,
			radii_options=None,
			radii_funcs=None
		):
		self.thresholds = thresholds
		self.radii_options = radii_options
		self.vals_to_clip = vals_to_clip
		self.vals_to_wrap = vals_to_wrap
		self.critical_regions = critical_regions
		self.model = model
		self.radii_funcs=radii_funcs
		self.continuous = continuous

		self.clip_mask = jnp.array([v is not None for v in self.vals_to_clip])
		self.clip_lower,self.clip_upper = self.separate_lower_and_upper_bounds_to_jnp_arrays(self.vals_to_clip)

		self.wrap_mask = jnp.array([v is not None for v in self.vals_to_wrap])
		self.wrap_lower,self.wrap_upper = self.separate_lower_and_upper_bounds_to_jnp_arrays(self.vals_to_wrap)

		if not jnp.all(~self.wrap_mask):
			raise ValueError("Currently doesn't faithfully support wrapping in action spheres - need to be able to deal with when wrapping causes the lower bound to be larger than the upper bound")
		
		self.distance_between_boxes = jax.vmap(
            lambda lb1, ub1, lb2, ub2: distance_from_box_to_box(lb1,ub1,lb2,ub2),
			in_axes=(0,0,None,None)
        )


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
	def generate_sphere(self,centre, radii):
		'''
		given a centre, radii for each dimension and lower/upper bounds for dimension=wise clippping/wrapping, produce the (irregular) sphere
		'''
		lb,ub = self.L_infinity(centre, radii)

		# deal with clipping
		lb_clipped = jnp.clip(lb,self.clip_lower, self.clip_upper)
		ub_clipped = jnp.clip(ub,self.clip_lower, self.clip_upper)

		# only apply clipping to the right places
		lb = jnp.where(self.clip_mask, lb_clipped, lb)
		ub = jnp.where(self.clip_mask, ub_clipped, ub)

		# TODO - deal with lower and upper bounds swapping after wrapping
		# NOTE - wrapping not fully supported yet so we don't use it at the moment... see the Value error in __init__...
		# deal with wrapping		
		# lb_wrapped = self.wrap_interval(lb,self.wrap_lower,self.wrap_upper)
		# ub_wrapped = self.wrap_interval(ub,self.wrap_lower,self.wrap_upper)

		# lb = jnp.where(self.wrap_mask, lb_wrapped, lb)
		# ub = jnp.where(self.wrap_mask, ub_wrapped, ub)

		# jax.debug.print('action centre: {}, clipped lb: {}, clipped ub: {}', centre, lb, ub)

		return lb,ub
	
	# generate the sphere size using a continuous function, parameterised by the distance to the nearest critical region, goal region or border
	# assuems we have a continuous function per action dimension
	def generate_sphere_continuous(self, action_centre, state=None, state_min=None, state_max=None):
		# make the box a single point if we are just working with 1 single state
		if state_min is None or state_max is None:
			state_min = state
			state_max = state

		# find the distance to the nearest critical region
		# find reachable set using a radius of 0
		frs_min,frs_max = self.model.step_set(state_min, state_max, action_centre, action_centre)
		

		# find minimum distance from the forward reachable set (with radius 0) to the critical regions
		frs_critical_distances = self.distance_between_boxes(
			self.critical_regions[:, 0, :],
			self.critical_regions[:, 1, :],
			frs_min,
			frs_max,
		)

		frs_closest_critical = jnp.min(frs_critical_distances)

		# calculate the sphere using the continuous function
		radii = jnp.array([f(frs_closest_critical) for f in self.radii_funcs])
		return self.generate_sphere(action_centre,radii)


	def generate_sphere_discrete(self, action_centre, state=None, state_min=None,state_max=None):
		'''
		for a given action, give the sphere - this is dependent on the current state.
		use state_min/max when calculating forward reachable sets, so calculating the action spheres for a set (box) of states

		This is the "public" facing function that we expect to get called elsewhere
		'''

		# make the box a single point if we are just working with 1 single state
		if state_min is None or state_max is None:
			state_min = state
			state_max = state

		# find reachable set using a radius of 0
		frs_min,frs_max = self.model.step_set(state_min, state_max, action_centre, action_centre)


		# find minimum distance from the forward reachable set (with radius 0) to the critical regions
		# find minimum distance from the forward reachable set (with radius 0) to the critical regions
		frs_critical_distances = self.distance_between_boxes(
			self.critical_regions[:, 0, :],
			self.critical_regions[:, 1, :],
			frs_min,
			frs_max,
		)

		frs_closest_critical = jnp.min(frs_critical_distances)

		# TODO - this could perhaps be sped up...
		# select the radii for each dimension
		mask = frs_closest_critical >= self.thresholds
		radii = jnp.where(
			mask[:, None],
			self.radii_options,
			0
		)

		# pick the largest radii values that are allowed
		idx = len(mask) - jnp.sum(mask)
		idx = jnp.maximum(idx,0)

		# jax.debug.print('closest is distance: {}, radii = {}', frs_closest_critical, radii[idx])

		return self.generate_sphere(action_centre,radii[idx])

	def get_action_sphere(self, action_centre, state=None, state_min=None, state_max = None):
		if self.continuous:
			return self.generate_sphere_continuous(action_centre,state,state_min,state_max)
		else:
			return self.generate_sphere_discrete(action_centre,state,state_min,state_max)
		
	# write the sphere definition to a file
	def __str__(self):
		if self.continuous:
			raise NotImplemented
		else:
			s = ''

			s += "thresholds = ["
			for threshold in self.thresholds:
				s += f'{threshold},'
			s += ']'

			s += '\n'

			s += 'radii = [\n' 
			for radii in self.radii_options:
				s += f'{radii},\n'
			s += ']'
			s += '\n\n'
		
			return s