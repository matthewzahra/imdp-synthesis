import RL.Reward
import RL.Evaluate_Secondary

'''
data structure to pair up the reward signal and then evaluation model for different optimsiations for the RL agent
'''


class RewardEval:
	def get_pair(self):
		return self.reward,self.evaluation
	

class GetCloseToArea(RewardEval):
	'''
	Reward/evaluate getting close to some box
	'''
	def __init__(self, region_lower, region_upper, dims = None):
		self.reward = RL.Reward.GetCloseToRegion(region_lower,region_upper,dims)
		self.evaluation = RL.Evaluate_Secondary.DistanceToRegion(region_lower,region_upper,dims)

class GetCloserThanBaseToArea(RewardEval):
	def __init__(self, region_lower, region_upper, dims = None):
		self.reward = RL.Reward.GetCloserToRegionThanPolicy(region_lower,region_upper,dims)
		self.evaluation = RL.Evaluate_Secondary.DistanceToRegion(region_lower,region_upper,dims)


class TimeTaken(RewardEval):
	'''
	Look to minimise or maximise the time taken
	'''
	def __init__(self,maximise_time=False):
		if maximise_time:
			self.reward = RL.Reward.OptimiseTimeSteps(1)
		else:
			self.reward = RL.Reward.OptimiseTimeSteps()
		self.evaluation = RL.Evaluate_Secondary.TimeSteps()

class ActionCosts(RewardEval):
	'''
	Tax the cost of actions
	'''
	def __init__(self,action_costs):
		self.reward = RL.Reward.AbsActionCost(action_costs)
		self.evaluation = RL.Evaluate_Secondary.EnergyEfficiency(action_costs)

class ActionSmoothness:
	def __init__(self):
		raise NotImplemented
	
class GetToRegionDoubleReward(RewardEval):
	'''
	we are using 2 types of reward here
	'''
	def __init__(self, region1_lower, region1_upper, region2_lower, region2_upper, dims = None):
		self.reward = RL.Reward.GetCloseToRegionAndBeatPolicy(region1_lower,region1_upper, region2_lower, region2_upper, dims)
		self.evaluation = RL.Evaluate_Secondary.DistanceToRegion(region2_lower,region2_upper,dims)