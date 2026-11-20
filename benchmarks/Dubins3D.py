import numpy as np

from benchmarks.models import DubinsDynamics3D
from core.box import Box
from core.rl.config import RLConfig


class Dubins3D(DubinsDynamics3D):
    '''
    Simplified version of the Dubin's vehicle benchmark, with a 3D state space and a 2D control input space.
    '''

    def __init__(self, args):
        DubinsDynamics3D.__init__(self, args)

        self.plot_dimensions = [0, 1]

        self.state_space = Box([-10, -10, -np.pi], [10, 10, np.pi], names=self.state_variables)
        self.state_grid = [100, 100, 100]
        self.input_space = Box([-0.5, -0.5], [0.5, 0.5])
        self.input_grid = [5, 5]

        region = self.state_space.restrict
        self.goal = [region(x=(-10, -5), y=(5, 10))]
        self.critical = [
            region(x=(-10, -1), y=(-1, 1)),
            region(x=(-1, 1), y=(-5, 5)),
        ]

        self.x0 = np.array([-7.5, -7.5, 0])

        # RL configuration: networks, PPO training, reward function, and the tube
        # grown around the RL rollouts to form the abstraction.
        self.rl_config = RLConfig(
            rl_algo='ppo',
            total_timesteps=5000000,
            proximity_dims=[0, 1],
            proximity_penalty=0.5,
            per_step_cost=0.05,
            RL_actions_per_state=9,
            inflation_rate=[(-5, 5), (-5, 5), (-5, 5)],
        )
