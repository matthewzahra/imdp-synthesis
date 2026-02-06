import numpy as np

'''
Define a way to evaluate secondary constraints so that we can test if the RL approach does better
'''

class EvaluateSecondary:
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
		self.action_scaling = action_scaling

	def get_score(self,state,action):
		return np.dot(np.abs(action),self.action_scaling)
	
class TimeSteps(EvaluateSecondary):
	'''
	Count number of time steps before we reach the goal
	'''

	def __init__(self):
		pass

	def get_score(self,state,action):
		return 1
	
class DistanceToRegion(EvaluateSecondary):
	'''
	Evaluate how close we get to a given region
	'''
	def __init__(self, region_lower, region_upper, dims = None):
		self.region_lower = region_lower
		self.region_upper = region_upper
		self.dims = dims

	def get_score(self,state,action):
		if self.dims: # only use selected dimenions of the state
			state = state[self.dims]

		closest_point = np.clip(state,self.region_lower,self.region_upper)

		distance = np.linalg.norm(state - closest_point)

		return distance