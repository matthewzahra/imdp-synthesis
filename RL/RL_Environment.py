import gymnasium as gym 
from gymnasium import spaces
import numpy as np

'''
Wrap the environment inside the gymnasium API so that SB3 plugs in easily.

NOTE: the RL agent will have access to all the actions possible, not just those in our derived set.
Therefore, we will project their output onto this set, and then apply the result to the system. 
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
			action_lower,
			action_upper,
			initial_state,
			model,
			policy_inputs,
			derive_set,
			reward_structure,
			partition,
			max_steps=1000
			):
		'''
		:param state_dim: dimension of the state space
		:param space_lower: lower corner of state space
		:param space_upper: upper corner of state space
		:param action_dim: dimension of the concrete action space
		:param action_lower: lower corner of the action space
		:param action_upper: upper corner of the action space
		:param initial_state: initial state for the agent
		:param model: the system model that implements the dynamics - must implement "step"
		:param policy_inputs: IMDP policy that maps states to a single concrete action
		:param derive_set: take a concrete action and derive the set of actions that we will allow the RL agent to choose from
		:param reward_structure: of type RL.Reward. Given the RL agent's suggested action and the resulting state it computes the reward
		:param partition: partition the state space into the goal and critical states so that the agent knows when it must terminate
		:param max_steps: max number of steps before we terminate the process
		'''


		# define the full action space
		self.action_space = spaces.Box(
			low=action_lower,
			high=action_upper,
			shape=action_dim,
			dtype=np.float32
		)

		# define the full state space
		self.observation_space = spaces.Box(
			low=space_lower,
			high=space_upper,
			shape=state_dim,
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

		assert initial_state not in partition.critical['idxs']

	# get a new episode
	def reset(self, seed=None, options=None):
		super().reset(seed=seed)
		self.state = self.initial_statex[k]
		return self.state, {} # TODO - do we need to do self.state.copy()?
	
	# Generate a single noise sample from the model
	# TODO - can we pre-compute these to speed this up?
	def _generate_noise(self):
		return np.random.multivariate_normal(
			mean=np.zeros(self.model.n),
			cov=self.model.noise['cov']**2 # TODO check this - we square it here as this is what we do in the MonteCarloSum class
		)
	
	# project an action outside of the allowed set back into it
	def _clip_action(self,action,constraints):
		return np.clip(
			action,
			constraints['lower'],
			constraints['upper']
		)

	# advnace the enviornment by 1 time step
	def step(self, proposed_action):

		# check if we have run for too long
		if self.t > self.max_steps:
			terminated = True
			reward = -100	# TODO - ADJUST  - minor penalty for not completing the task in time
			info = {}
			return self.state, reward, terminated, info

		self.t += 1
		noise = self._generate_noise()

		# clip action into the valid set
		policy_action = self.policy_inputs[self.state]
		action_set = self.derive_set(policy_action)
		clipped_action = self._clip_action(proposed_action,action_set) # TODO - this will break at the moment until we decide how the derive_set is meant to work...

		# progress the state using the model's dynamics 
		new_state = self.model.step(self.state,clipped_action,noise)
		self.state = new_state

		# find what abstract state we are in 
		abstract_state = self.partition.x2state(new_state)

		# check if we are in a critical region
		if abstract_state in self.partition.critical['idxs']:
			terminated = True
			reward = -1000		# TODO - pick this value a bit better...{}

		# check if we are in a goal state
		else:
			if abstract_state in self.partition.goal['idxs']:
				terminated = True 
			reward = self.reward_structure.getReward(state=new_state, action=clipped_action)



		truncated = False	# use for ending a run earlier that could have in theory continue
		info = {}

		return new_state, reward, terminated, truncated, info
	
	# for visualization
	def render(self):
		...

	# cleanup resources
	def close(self):
		...
	