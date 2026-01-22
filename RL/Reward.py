import numpy as np

'''
Define the different reward structures
'''

class RewardStructure:
	def getReward(self,state,action):
		raise NotImplemented


class MinDistanceToGoal(RewardStructure):
	'''
	Always look to minimise the distance to the goal
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
		return np.linalg.norm(delta) * self.scaling
	
class ActionCost(RewardStructure):
	def __init__(
			self,
			action_costs 	# a dictionary with a multiplier for ea
			):
		...