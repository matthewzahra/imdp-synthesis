import numpy as np

from benchmarks.models import DubinsDynamics4D
from core.box import Box
from core.rl.config import RLConfig


class Dubins4D(DubinsDynamics4D):
    '''
    Dubin's vehicle benchmark, with a 4D state space and a 2D control input space.
    '''

    def __init__(self, args):
        DubinsDynamics4D.__init__(self, args)

        self.plot_dimensions = [0, 1]

        self.state_space = Box([-10, 0, -np.pi, self.v_min], [10, 10, np.pi, self.v_max], names=self.state_variables)
        self.state_grid = [100, 50, 100, 100]
        self.input_space = Box([-0.5, -0.5], [0.5, 0.5])
        self.input_grid = [5, 5]

        region = self.state_space.restrict
        self.goal = [region(x=(6, 9), y=(6, 9))]
        self.critical = [
            region(x=(4, 5), y=(5, 10), angle=(-2 * np.pi, 2 * np.pi)),
            region(x=(-1, 0), y=(0, 5), angle=(-2 * np.pi, 2 * np.pi)),
            region(x=(-5, -1), y=(4, 5), angle=(-2 * np.pi, 2 * np.pi)),
        ]

        self.x0 = np.array([-3, 2, np.pi/2, 0.0])

        # RL configuration: networks, PPO training, reward function, and the tube
        # grown around the RL rollouts to form the abstraction.
        self.rl_config = RLConfig(
            rl_algo='ppo',
            total_timesteps=20000000,
            proximity_dims=[0, 1],
            proximity_penalty=0.5,
            per_step_cost=0.05,
            RL_actions_per_state=9,
            inflation_rate=[(-5, 5), (-5, 5), (-5, 5), (-5, 5)],
        )
