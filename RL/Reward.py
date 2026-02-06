import numpy as np
from typing import Optional

'''
Define the different reward structures
'''

class RewardStructure:
	def getReward(self,state,action):
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

	def getReward(self,state,action):
		box_min, box_max = self.goal_box
		delta = np.maximum(0, np.maximum(box_min - state, state - box_max))
		return -1*(np.linalg.norm(delta) * self.scaling) # make negative since we are trying to maximise reward 
	
class AbsActionCost(RewardStructure):
	def __init__(
			self,
			action_costs 	# a dictionary with a multiplier for each dimension in the action space
		):
		self.action_costs = action_costs
	
	def getReward(self, state, action):
		'''
		we will make all the values their absolute values in the action
		'''
		return np.dot(np.abs(action), self.action_costs) ** 3 # we cube it to avoid saturation at the boundaries - we either want to reward big actions or small ones
	
class SmoothMovements(RewardStructure):
	'''
	Penalise Jerky movements of consecutive actions
	'''
	def __init__(
		self,
		action_element_scalings	# scalings for each dimension of the actions
	):
		self.action_element_scalings = action_element_scalings
		self.prev_action = None

	def getReward(self,state,action):
		if not self.prev_action:
			self.prev_action = action
			return 0
		
		else:
			reward = np.dot(self.action_element_scalings, np.abs(action - self.prev_action))
			self.prev_action = action
			return reward
		
class OptimiseTimeSteps(RewardStructure):
	'''
	Look to reward taking a long or a short time to reach the goal state
	'''
	def __init__(self, time_step_reward=-1):
		self.time_step_reward = time_step_reward

	def getReward(self, state, action):
		return self.time_step_reward
	
class GetCloseToRegion(RewardStructure):
	'''
	Reward getting the state close to a given state.

	Can specify what elements of the state vectors to consider
	'''

	def __init__(self, target_min, target_max, dims: Optional[list[int]] = None):
		self.target_min = target_min
		self.target_max = target_max
		self.dims = dims

	def getReward(self, state, action):
		if self.dims: # only use selected dimenions of the state
			state = state[self.dims]

		closest_point = np.clip(state,self.target_min,self.target_max)

		distance = np.linalg.norm(state - closest_point)

		return -1 * distance

		