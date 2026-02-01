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