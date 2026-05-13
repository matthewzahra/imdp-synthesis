import RL.Reward
import RL.Evaluate_Secondary
import numpy as np

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

class MaxActionCosts(RewardEval):
	def __init__(self,action_costs):
		self.reward = RL.Reward.MaxAbsActionCost(action_costs)
		self.evaluation = RL.Evaluate_Secondary.EnergyEfficiency(action_costs)		

class ActionSmoothness(RewardEval):
	def __init__(self, action_scaling_reward, action_scaling_evaluate):
		self.reward = RL.Reward.SmoothMovements(action_scaling_reward)
		self.evaluation = RL.Evaluate_Secondary.ActionSmoothness(action_scaling_evaluate)

class JerkyMovements(RewardEval):
	def __init__(self, action_scaling_reward, action_scaling_evaluate):
		self.reward = RL.Reward.JerkyMovements(action_scaling_reward)
		self.evaluation = RL.Evaluate_Secondary.ActionSmoothness(action_scaling_evaluate)
	
class GetToRegionDoubleReward(RewardEval):
	'''
	we are using 2 types of reward here
	'''
	def __init__(self, region1_lower, region1_upper, region2_lower, region2_upper, dims = None):
		self.reward = RL.Reward.GetCloseToRegionAndBeatPolicy(region1_lower,region1_upper, region2_lower, region2_upper, dims)
		self.evaluation = RL.Evaluate_Secondary.DistanceToRegion(region2_lower,region2_upper,dims)


'''
generate the reward/evaluation pairs
'''

def generate_reward_eval(model_name):
	reward_evals = dict()

	# Dubins_small
	reward_evals['Dubins_small'] = dict()
	# reward_evals['Dubins_small']['get_closer_than_base_to_bottom_right'] = RL.Reward_Evaluate.GetCloserThanBaseToArea(region_lower=np.array([10,-10]), region_upper=np.array([10,-10]), dims=[0,1])
	# reward_evals['Dubins_small']['get_closer_than_base_to_bottom_opening'] = RL.Reward_Evaluate.GetCloserThanBaseToArea(region_lower=np.array([-1,-6.75]), region_upper=np.array([-1,-6.75]), dims=[0,1])
	reward_evals['Dubins_small']['get_to_bottom_right'] = RL.Reward_Evaluate.GetCloseToArea(region_lower=np.array([10,-10]), region_upper=np.array([10,-10]), dims=[0,1])
	reward_evals['Dubins_small']['get_close_to_bottom_opening'] = RL.Reward_Evaluate.GetCloseToArea(region_lower=np.array([-1,-6.75]), region_upper=np.array([-1,-6.75]), dims=[0,1])
	# reward_evals['Dubins_small']['bottom_opening_double_reward'] = RL.Reward_Evaluate.GetToRegionDoubleReward(region1_lower=np.array([-1,-6.75]), region1_upper=np.array([-1,-6.75]), region2_lower=np.array([10,-10]), region2_upper=np.array([10,-10]), dims=[0,1])
	# reward_evals['Dubins_small']['bottom_opening_double_reward2'] = RL.Reward_Evaluate.GetToRegionDoubleReward(region2_lower=np.array([-1,-6.75]), region2_upper=np.array([-1,-6.75]), region1_lower=np.array([10,-10]), region1_upper=np.array([10,-10]), dims=[0,1])
	reward_evals['Dubins_small']['smooth_actions'] = ActionSmoothness(action_scaling_reward=np.array([1,1]),action_scaling_evaluate=np.array([1,1]))
	reward_evals['Dubins_small']['energy_efficient'] = ActionCosts(action_costs=[1,1]) # TODO - should we tax the angle?
	reward_evals['Dubins_small']['jerky_actions'] = JerkyMovements(action_scaling_reward=np.array([1,1]),action_scaling_evaluate=np.array([1,1]))
	reward_evals['Dubins_small']['max_energy'] = MaxActionCosts(action_costs=[1,1])
	# reward_evals['Dubins_small']['energy_efficient_no_angle_tax'] = ActionCosts(action_costs=[0,1])

	# MountainCar
	reward_evals['MountainCar'] = dict()
	reward_evals['MountainCar']['energy_efficient'] = ActionCosts(action_costs=[1])
	reward_evals['MountainCar']['action_smoothness'] = ActionSmoothness(action_scaling_reward=np.array([1]),action_scaling_evaluate=np.array([1]))
	reward_evals['MountainCar']['jerky_actions'] = JerkyMovements(action_scaling_reward=np.array([1]),action_scaling_evaluate=np.array([1]))
	reward_evals['MountainCar']['max_energy'] = MaxActionCosts(action_costs=[1])

	# Drone2D
	reward_evals['Drone2D'] = dict()	
	reward_evals['Drone2D']['smooth_actions'] = ActionSmoothness(action_scaling_reward=np.array([1,1]),action_scaling_evaluate=np.array([1,1]))
	reward_evals['Drone2D']['energy_efficient'] = ActionCosts(action_costs=[1,1])
	reward_evals['Drone2D']['jerky_actions'] = JerkyMovements(action_scaling_reward=np.array([1,1]),action_scaling_evaluate=np.array([1,1]))
	reward_evals['Drone2D']['max_energy'] = MaxActionCosts(action_costs=[1,1])

	# Pendulum
	reward_evals['Pendulum'] = dict()
	reward_evals['Pendulum']['smooth_actions'] = ActionSmoothness(action_scaling_reward=np.array([1]),action_scaling_evaluate=np.array([1]))
	reward_evals['Pendulum']['energy_efficient'] = ActionCosts(action_costs=[1])
	reward_evals['Pendulum']['jerky_actions'] = JerkyMovements(action_scaling_reward=np.array([1]),action_scaling_evaluate=np.array([1]))
	reward_evals['Pendulum']['max_energy'] = MaxActionCosts(action_costs=[1])

	return reward_evals[model_name]