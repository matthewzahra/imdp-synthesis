import gymnasium as gym 
from gymnasium import spaces
import numpy as np
from RL.helper_functions import project_action

'''
Wrap the environment inside the gymnasium API so that SB3 plugs in easily.

NOTE: RL agent will only be allowed ot suggest actions in the hyperrectangle [(-1,...,-1), (1,...,1)] - we will then produce a concrete action that is scaled appropriately. 
The origin is the same as taking the centre of the allowed sphere.

TODO - do we want to give the agent the centre of the sphere as part of the state that it sees???
'''

# TODO - needs to be able to handle finite and infinite horizons - currently we assume it is an INFINITE horizon
# TODO - currently, if we hit a critical region we terminate and give a large negative reward. There are some thing we can do with wrappers to better enforce hard constraints if this doesn't work

class Env(gym.Env):
	def __init__(
			self,
			state_dim,
			space_lower,
			space_upper,
			action_dim,
			initial_state,
			model,
			policy_inputs,
			derive_set,
			reward_structure,
			partition,
			max_steps=200
			):
		'''
		:param state_dim: dimension of the state space
		:param space_lower: lower corner of state space
		:param space_upper: upper corner of state space
		:param action_dim: dimension of the concrete action space
		:param initial_state: initial state for the agent
		:param model: the system model that implements the dynamics - must implement "step"
		:param policy_inputs: IMDP policy that maps states to a single concrete action
		:param derive_set: take a concrete action and derive the set of actions that we will allow the RL agent to choose from
		:param reward_structure: of type RL.Reward. Given the RL agent's suggested action and the resulting state it computes the reward
		:param partition: partition the state space into the goal and critical states so that the agent knows when it must terminate
		:param max_steps: max number of steps before we terminate the process
		'''


		# define the full action space as the hyperrectangle [(-1,...,-1), (1,...,1)]
		self.action_space = spaces.Box(
			low=np.asarray([-1 for _ in range(action_dim)], dtype=np.float32),
			high=np.asarray([1 for _ in range(action_dim)], dtype=np.float32),
			shape=(action_dim,),
			dtype=np.float32
		)

		# define the full state space
		self.observation_space = spaces.Box(
			low=np.asarray(space_lower, dtype=np.float32),
			high=np.asarray(space_upper, dtype=np.float32),
			shape=(state_dim,),
			dtype=np.float32
		)

		# the current state
		self.initial_state = initial_state
		self.state = initial_state

		# the current time step
		self.t = 0
		self.max_steps = max_steps 

		# the model that implements the dynamics
		self.model = model

		self.derive_set = derive_set
		self.policy_inputs = policy_inputs
		self.reward_structure = reward_structure
		self.partition = partition

		assert partition.x2state(initial_state)[0] not in partition.critical['idxs']

		self.too_long = 0
		self.goal_count = 0
		self.critical_count = 0

	# get a new episode
	def reset(self, seed=None, options=None):
		super().reset(seed=seed)
		self.state = self.initial_state
		self.t = 0
		return self.state, {} 
	
	# Generate a single noise sample from the model
	def _generate_noise(self):
		return np.random.multivariate_normal(
			mean=np.zeros(self.model.n),
			cov=self.model.noise['cov']**2 # TODO check this - we square it here as this is what we do in the MonteCarloSum class
		)
	
	# TODO - check that this is correct ...
	# given an action proposed in the hyperrectangle [(-1,...,-1), (1,...,1)], find the corresponding real concrete action by scaling appropriately
	def _project_action(self, action, action_lower, action_upper):
		return project_action(action, action_lower, action_upper)

	# advance the enviornment by 1 time step
	def step(self, proposed_action):

		terminated = False
		truncated = False	# use for ending a run earlier that could have in theory continue
		info = {}

		# check if we have run for too long
		if self.t > self.max_steps:
			self.too_long += 1
			terminated = True
			reward = -100	# minor penalty for not completing the task in time
			info = {}
			return self.state, reward, np.array(terminated, dtype=bool), np.array(truncated, dtype=bool), info

		self.t += 1
		noise = self._generate_noise()

		# project action into the current action sphere
		policy_action = self.policy_inputs[self.partition.x2state(self.state)[0]]
		# with open('policy_actions', 'a') as f:
		# 	f.write(f"{policy_action}\n")

		action_set_lower_bounds, action_set_upper_bounds = self.derive_set(policy_action)
		# clipped_action = self._clip_action(proposed_action,action_set_lower_bounds,action_set_upper_bounds) # TODO - this will break at the moment until we decide how the derive_set is meant to work...
		projected_action = self._project_action(proposed_action,action_set_lower_bounds,action_set_upper_bounds)

		# progress the state using the model's dynamics 
		new_state = self.model.step(self.state,projected_action,noise)
		self.state = new_state

		# find what abstract state we are in 
		abstract_state = self.partition.x2state(new_state)[0]

		# check if we are in a critical region
		if abstract_state in self.partition.critical['idxs']:
			self.critical_count += 1
			terminated = True
			reward = -1000 # large penalty for entering a critical region

		# check if we are in a goal state
		else:
			if abstract_state in self.partition.goal['idxs']:
				self.goal_count += 1
				terminated = True 
			reward = self.reward_structure.getReward(state=new_state, action=projected_action)


		return new_state, reward, np.array(terminated, dtype=bool), np.array(truncated, dtype=bool), info
	
	# for visualization
	def render(self):
		...

	# cleanup resources
	def close(self):
		...
	