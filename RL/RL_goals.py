import RL
import numpy as np

'''
generate the reward/evaluation pairs
'''

def generate_reward_eval():
	reward_evals = dict()

	# reward_evals['minimise_action_costs'] = RL.Reward_Evaluate.ActionCosts(np.array([0,-1])) # use 0 to not tax the angle in the input
	# reward_evals['maximise_action_costs'] = RL.Reward_Evaluate.ActionCosts(np.array([0,1])) # use 0 to not tax the angle in the input
	# reward_evals['get_close_top_right'] = RL.Reward_Evaluate.GetCloseToArea(region_lower=np.array([10,10]), region_upper=np.array([10,10]), dims=[0,1])
	# reward_evals['get_close_vertical_critical'] = RL.Reward_Evaluate.GetCloseToArea(region_lower=np.array([-1,-5]), region_upper=np.array([1,4]), dims=[0,1])
	# reward_evals['get_closer_than_base_to_bottom_right'] = RL.Reward_Evaluate.GetCloserThanBaseToArea(region_lower=np.array([10,-10]), region_upper=np.array([10,-10]), dims=[0,1])
	# reward_evals['get_closer_than_base_to_top_right'] = RL.Reward_Evaluate.GetCloserThanBaseToArea(region_lower=np.array([10,10]), region_upper=np.array([10,10]), dims=[0,1])
	# reward_evals['get_closer_than_base_to_bottom_left'] = RL.Reward_Evaluate.GetCloserThanBaseToArea(region_lower=np.array([-10,-10]), region_upper=np.array([-10,-10]), dims=[0,1])
	# reward_evals['get_closer_than_base_to_vertical_critical'] = RL.Reward_Evaluate.GetCloserThanBaseToArea(region_lower=np.array([-1,-5]), region_upper=np.array([1,4]), dims=[0,1])
	# reward_evals['get_closer_than_base_to_top_opening'] = RL.Reward_Evaluate.GetCloserThanBaseToArea(region_lower=np.array([-1,6.5]), region_upper=np.array([-1,6.5]), dims=[0,1])
	# reward_evals['top_opening_double_reward'] = RL.Reward_Evaluate.GetToRegionDoubleReward(region1_lower=np.array([-1,6.5]), region1_upper=np.array([-1,6.5]), region2_lower=np.array([10,10]), region2_upper=np.array([10,10]), dims=[0,1])
	# reward_evals['top_opening_double_reward2'] = RL.Reward_Evaluate.GetToRegionDoubleReward(region2_lower=np.array([-1,5]), region2_upper=np.array([1,10]), region1_lower=np.array([10,10]), region1_upper=np.array([10,10]), dims=[0,1])
	reward_evals['smooth_actions'] = RL.Reward_Evaluate.ActionSmoothness(action_scaling_reward=np.array([1,1]),action_scaling_evaluate=np.array([1,1]))

	return reward_evals