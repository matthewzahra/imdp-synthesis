import numpy as np
from typing import Optional
import jax.numpy as jnp

'''
Define the different reward structures
'''

class RewardStructure:
	def __init__(self, include_prev_action=False):
		self.include_prev_action = include_prev_action

	def getReward(self,old_state, new_base_state, new_rl_state, policy_action, rl_action):
		raise NotImplemented


class MinDistanceToGoal(RewardStructure):
	'''
	Always look to minimise the distance to the goal.

	NOTE: assumes a single goal region!
	'''
	def __init__(
			self,
			goal_box,		# assumed to be an axis-aligned hyperrectangle in teh form [min,max]
			scaling = 1
		):
		super().__init__()
		self.goal_box = goal_box
		self.scaling = scaling

	def getReward(self,old_state, new_base_state, new_rl_state, policy_action, rl_action):
		box_min, box_max = self.goal_box
		delta = np.maximum(0, np.maximum(box_min - new_rl_state, new_rl_state - box_max))
		return -1*(np.linalg.norm(delta) * self.scaling) # make negative since we are trying to maximise reward 
	
class AbsActionCost(RewardStructure):
	def __init__(
			self,
			action_costs 	# a dictionary with a multiplier for each dimension in the action space
		):
		super().__init__()
		self.action_costs = action_costs
	
	def getReward(self, old_state, new_base_state, new_rl_state, policy_action, rl_action):
		'''
		we will make all the values their absolute values in the action
		'''
		return np.dot(np.abs(rl_action), self.action_costs) ** 3 # we cube it to avoid saturation at the boundaries - we either want to reward big actions or small ones
	
class SmoothMovements(RewardStructure):
	'''
	Penalise Jerky movements of consecutive actions
	'''
	def __init__(
		self,
		action_element_scalings	# scalings for each dimension of the actions
		):
		super().__init__(include_prev_action=True)	# we want to include the previous action in the RL agent's state 
		self.action_element_scalings = action_element_scalings
		self.prev_action = np.zeros(self.action_element_scalings.size)	# initially the previous action is (0,...,0)

	def getReward(self,old_state, new_base_state, new_rl_state, policy_action, rl_action):
		reward = np.dot(self.action_element_scalings, np.abs(rl_action - self.prev_action))
		self.prev_action = rl_action
		return -1 * reward ** 2		# square so that large adjustments are penalised more
	
class SmoothMovementsAveraged(RewardStructure):
	'''
	Penalise Jerky movements, but average over a trace length so that we don't take the trace length into account
	'''

	# TODO - Not sure this works?
	def __init__(self):
		raise NotImplemented
		
class OptimiseTimeSteps(RewardStructure):
	'''
	Look to reward taking a long or a short time to reach the goal state
	'''
	def __init__(self, time_step_reward=-1):
		super().__init__()
		self.time_step_reward = time_step_reward

	def getReward(self, old_state, new_base_state, new_rl_state, policy_action, rl_action):
		return self.time_step_reward
	
class GetCloseToRegion(RewardStructure):
	'''
	Reward getting the state close to a given state.

	Can specify what elements of the state vectors to consider
	'''

	def __init__(self, target_min, target_max, dims: Optional[list[int]] = None):
		super().__init__()
		self.target_min = target_min
		self.target_max = target_max
		self.dims = dims

	def getReward(self, old_state, new_base_state, new_rl_state, policy_action, rl_action):
		if self.dims: # only use selected dimenions of the state
			new_rl_state = new_rl_state[jnp.array(self.dims)]

		closest_point = np.clip(new_rl_state,self.target_min,self.target_max)

		distance = np.linalg.norm(new_rl_state - closest_point)

		return -1 * (distance ** 2) / 25

class GetCloserToRegionThanPolicy(RewardStructure):
	def __init__(self, target_min, target_max, dims: Optional[list[int]] = None):
		super().__init__()
		self.target_min = target_min
		self.target_max = target_max
		self.dims = dims

	def getReward(self, old_state, new_base_state, new_rl_state, policy_action, rl_action):
		if self.dims: # only use selected dimenions of the state
			new_base_state = new_base_state[jnp.array(self.dims)]
			new_rl_state = new_rl_state[jnp.array(self.dims)]

		closest_point_base = np.clip(new_base_state,self.target_min,self.target_max)
		closest_point_rl = np.clip(new_rl_state,self.target_min,self.target_max)

		rl_distance = np.linalg.norm(new_rl_state - closest_point_rl)
		base_distance = np.linalg.norm(new_base_state - closest_point_base)

		return base_distance**2 - rl_distance**2

class GetCloseToRegionAndBeatPolicy(RewardStructure):
	def __init__(self, target1_min, target1_max, target2_min, target2_max, dims: Optional[list[int]] = None):
		super().__init__()
		self.beat_policy = GetCloserToRegionThanPolicy(target1_min, target1_max, dims)
		self.get_to_region = GetCloseToRegion(target2_min, target2_max, dims)

	def getReward(self, old_state, new_base_state, new_rl_state, policy_action, rl_action):
		return self.beat_policy.getReward(old_state, new_base_state, new_rl_state, policy_action, rl_action) + self.get_to_region.getReward(old_state, new_base_state, new_rl_state, policy_action, rl_action)