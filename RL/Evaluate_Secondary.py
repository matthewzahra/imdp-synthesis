import numpy as np
from core.helper_functions import get_cell_index

'''
Define a way to evaluate secondary constraints so that we can test if the RL approach does better
'''

class EvaluateSecondary:
	def __init__(self, include_prev_action=False):
		self.include_prev_action=include_prev_action


	def get_score(self,state,action) -> float:
		'''
		NOTE: given the CURRENT CONCRETE action and the CURRENT CONCRETE state, generate a score. 
		# TODO - should we use the NEW CONCRETE state instead???
		'''

		raise NotImplemented
	
class EnergyEfficiency(EvaluateSecondary):
	'''
	We simply scale the absolute value of each action by some scalar.
	We essentially just perform the dot product
	'''
	def __init__(self,action_scaling):
		super().__init__()
		self.action_scaling = action_scaling

	def get_score(self,state,action):
		return np.dot(np.abs(action),self.action_scaling)
	
class TimeSteps(EvaluateSecondary):
	'''
	Count number of time steps before we reach the goal
	'''

	def __init__(self):
		super().__init__()

	def get_score(self,state,action):
		return 1
	
class DistanceToRegion(EvaluateSecondary):
	'''
	Evaluate how close we are to a given region at each time step. 
	Also track the closest we got to it.
	'''
	def __init__(self, region_lower, region_upper, dims = None):
		super().__init__()
		self.region_lower = region_lower
		self.region_upper = region_upper
		self.dims = dims
		self.closest = float('inf')

	def get_score(self,state,action):
		if self.dims: # only use selected dimenions of the state
			state = state[self.dims]

		closest_point = np.clip(state,self.region_lower,self.region_upper)

		distance = np.linalg.norm(state - closest_point)

		self.closest = min(self.closest,distance)

		return distance
	
class ActionSmoothness(EvaluateSecondary):
	'''
	Evaluate how smooth the actions are 
	'''
	def __init__(self,action_scaling):
		super().__init__(include_prev_action=True)
		self.action_scaling = action_scaling
		self.prev_action = np.zeros(self.action_scaling.size)

	def get_score(self,state,action):
		score = np.dot(self.action_scaling, np.abs(action - self.prev_action))
		self.prev_action = action
		return score

class MatrixNorm(EvaluateSecondary):
	'''
	Use the energy norm (squared)
	'''
	def __init__(self,Q,R,centers,target_cells):
		self.Q = Q
		self.R = R
		self.centers = centers
		self.target_cells = target_cells

	# TODO - also be able to return state_cost and action_cost as well as their sum
	def get_score(self,state,action):
		target_cell = self.target_cells[get_cell_index(self.centers,state)]
		state_cost = (target_cell - state)@self.Q@(target_cell - state)
		action_cost = action@self.R@action
		return state_cost + action_cost 