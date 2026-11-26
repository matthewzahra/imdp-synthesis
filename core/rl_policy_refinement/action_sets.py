import jax.numpy as jnp
from core.rl_policy_refinement.helper_functions import distance_from_box_to_box

'''
here we define the structures that help us derive our action sets
NOTE: we abuse the word "sphere" to mean n-dimensional shape - they are n-spheres in the simplest case but do not need to have the same radius in each dimension
'''

class ActionSphere:
	def __init__(
			self,
			vals_to_clip):
		self.clip_lower, self.clip_upper = self._separate_lower_and_upper_bounds_to_jnp_arrays(vals_to_clip)


	def _separate_lower_and_upper_bounds_to_jnp_arrays(self,bounds):
		lower, upper = bounds

		lower = jnp.array(lower) if lower is not None else -jnp.inf
		upper = jnp.array(upper) if upper is not None else jnp.inf
		return lower, upper

	# L_infinity ball around a point clipped to lower and upper bounds so that we stay within the allowed action space
	def clipped_L_infinity(self, centre, radii):
		'''
		:param centre: centre coordinates of the ball
		:param radii: radius for each dimension - it does not need to be the same everywhere. Assumes all values are non-negative!!!

		return: lower bounds, upper bounds
		'''
		assert centre.size == radii.size
		lb, ub = centre-radii, centre+radii

		return jnp.maximum(lb,self.clip_lower), jnp.minimum(ub,self.clip_upper)

	# NOTE: implement by inheriting classes
	def generate_action_sphere(self):
		raise NotImplemented

class DiscreteFixedSphere(ActionSphere):
	'''
	class representing a sphere fixed in size at all times
	'''
	def __init__(
			self,
			vals_to_clip,
			model,
			radii):

		super().__init__(vals_to_clip)

		self.model = model
		self.radii = radii

	def get_action_sphere(self,action_centre, state_min ,state_max):
		return super().clipped_L_infinity(action_centre, self.radii)

class DiscreteCriticalRegionAwareSphere(ActionSphere):
	'''
	class representing a sphere that simulates a model step, and then chooses its according
	to a step-wise function that takes input as distance to nearest critical region and 
	returns the action set
	'''

	def __init__(
			self,
			vals_to_clip,
			critical_regions,
			model,
			radii
	):
		super().__init__(vals_to_clip)
		self.vals_to_clip = vals_to_clip
		self.critical_regions = critical_regions
		self.model = model
		self.radii_options = radii
		raise ValueError("Need to implement 'create_borders'...")

	def gen_action_sphere(self, action_centre, state_min, state_max):
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

		return super().clipped_L_infinity(action_centre, radii[idx])