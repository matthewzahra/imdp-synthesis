from RL.Reward_Evaluate import RewardEval
from stable_baselines3.common.vec_env import VecNormalize, VecMonitor
from stable_baselines3 import SAC
from RL.NeuralNetworks import ZeroActorLastLayer

'''
frame work to train several agents and then simulate them all
'''

class Agents():
	def __init__(self,reward_evals: dict[str,RewardEval], init_env, timesteps: int = 10_000):
		'''
		Docstring for __init__
		
		:param self: Description
		:param reward_evals: Description
		:type reward_evals: dict[str, RewardEval]
		:param init_env: should accept a reward_structure and return an initialised environment for training the RL agent
		:param timesteps: Description
		:type timesteps: int
		'''
		self.reward_evals = reward_evals
		self.init_env = init_env
		self.timesteps = timesteps

		for k,v in self.reward_evals.items():
			self.reward_evals[k] = v.get_pair()


	def train_agents(self, dont_train=False):
		for s,(reward_structure,evaluation) in self.reward_evals.items():
			env = VecMonitor(self.init_env(reward_structure))

			# NOTE - that since we normalize the observartions, we must do so again when we used the trained RL agent to predict
			# normalize the observations
			env = VecNormalize(
				env,
				norm_obs=True,
				norm_reward=False,
				clip_obs=10.0
			)

			if not dont_train:
				agent = SAC(
					ZeroActorLastLayer,
					env,
					verbose=0,
					ent_coef="auto_0.1",           # TODO - should play with this: disable entropy otherwise the agent collapses on smaller actions where possible. Great if we want minimization, poor if we want maximisation
					target_entropy="auto",
					tensorboard_log="./sac_logs/",
					policy_kwargs=dict(net_arch=[256, 256]),
					learning_starts=20_000
				)

				# train the agent 
				print(f"Training agent for {s}")
				agent.learn(total_timesteps=self.timesteps, progress_bar=True)

				print(f"Took too long: {env.envs[0].unwrapped.too_long}")
				print(f"Got to Goal State: {env.envs[0].unwrapped.goal_count}")
				print(f"Hit the critical state: {env.envs[0].unwrapped.critical_count}")

				print("Saving Agent")
				# save the trained agent
				agent.save(f'RL/agents/sac_agent_{s}_{self.timesteps}')

			print("Saving VecNormalize statistics")
			env.save(f"RL/agent_envs/vecnormalize_{s}.pkl")


	def get_agents_envs_evals(self):
		# return the agents, their enviornments and the evaluation structures
		agent_envs = []

		for s,(reward_structure,evaluation) in self.reward_evals.items():
			agent = SAC.load(f"RL/agents/sac_agent_{s}_{self.timesteps}")
			dummy_env = self.init_env(reward_structure)
			vecnorm = VecNormalize.load(f"RL/agent_envs/vecnormalize_{s}.pkl", dummy_env)
			vecnorm.training = False # don't allow the saved statistics to update
			vecnorm.norm_reward = False # don't normalise the rewards
		
			agent_envs.append((s,agent,vecnorm,evaluation))

		return agent_envs
