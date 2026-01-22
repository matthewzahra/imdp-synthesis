# class for the Reinforcement Learning Agent
# needs to train before we can use step
class RL_Agent:
	def __init__(self, model):
		'''
		Docstring for __init__
		
		:param model: from benchmark.models, this is the model of the system we want to train on - it contains all the dynamics
		'''
		self.model = model


	def train(self):
		pass

	# assumes that self.train() has already been executed
	def step(self,state,action):
		pass