
from benchmarks.models import DubinsDynamics3D
import numpy as np

from core.rl.config import RLConfig


class Dubins3D(DubinsDynamics3D):
    '''
    Simplified version of the Dubin's vehicle benchmark, with a 3D state space and a 2D control input space.
    '''

    def __init__(self, args):
        DubinsDynamics3D.__init__(self, args)
        
        self.plot_dimensions = [0, 1]

        self.set_spec()

    def set_spec(self):
        '''
        Set the abstraction parameters and the reach-avoid specification.
        '''

        self.partition = {}

        # Authority limit for the control u, both positive and negative
        self.uMin = [-0.5, -0.5]
        self.uMax = [0.5, 0.5]
        self.num_actions = [5, 5]

        self.partition['boundary'] = np.array([[-10, -10, -np.pi], [10, 10, np.pi]])
        self.partition['number_per_dim'] = np.array([100, 100, 100])

        self.goal = np.array([
            [[-10, 5, -np.pi], [-5, 10, np.pi]]
        ], dtype=float)

        self.critical = np.array([
            [[-10, -1, -np.pi], [-1, 1, np.pi]],
            [[-1, -5, -np.pi], [1, 5, np.pi]]
        ], dtype=float)

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

        return
