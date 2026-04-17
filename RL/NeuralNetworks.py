'''
Define some custome Neural Networks here for use by the RL algorithms
'''

from stable_baselines3.sac.policies import SACPolicy
from torch import nn

class ZeroActorLastLayer(SACPolicy):
	'''
	identical to the standard NN used in SB3's SAC algorithm, except we initialise the actor's last layer to be all 0s
	'''

	def _build(self, lr_schedule):
		# build the actor and critic networks as normal
		super()._build(lr_schedule=lr_schedule)

		# if the actor network is the linear layer that we expect, we want to 0 it
		# we remember that it follows a Guassian distribution, so we set the average (mu) to 0 

		# the network looks to output mu and log_std
		if isinstance(self.actor.mu, nn.Linear):
			nn.init.constant_(self.actor.mu.weight, 0.0)
			nn.init.constant_(self.actor.mu.bias, 0.0)

			
		# we look to set the variance to 0. We set its average to 0 and then its bias to a very negative number - we remember that we are considering its log
		# if we set its bias to be 0 we set the std to 1 essentially, which is a lot of noise
		if isinstance(self.actor.log_std, nn.Linear):
			nn.init.constant_(self.actor.log_std.weight, 0.0)
			nn.init.constant_(self.actor.log_std.bias, -20)
