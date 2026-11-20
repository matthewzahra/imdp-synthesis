import numpy as np

from benchmarks.models import AffineDynamics
from core.box import Box
from core.rl.config import RLConfig


class Test1D(AffineDynamics):
    '''
    Simple 1D test benchmark (x' = x + u + w), with a 1D state space and a 1D control input space.
    '''

    def __init__(self, args):
        AffineDynamics.__init__(self, args, A=[[1]], B=[[1]], q=[0],
                                gaussian=dict(variance=np.array([0.2])**2, cells=[10]),
                                triangular=dict(halfwidth=[0.2], cells=[10]),
                                state_variables=['position'])

        self.plot_dimensions = [0]

        self.state_space = Box([-5], [5], names=self.state_variables)
        self.state_grid = [10]
        self.input_space = Box([-1], [1])
        self.input_grid = [3]

        self.goal = [self.state_space.restrict(position=(-1, 1))]
        self.critical = []

        self.x0 = np.array([-2.5])

        # RL configuration: networks, PPO training, reward function, and the tube
        # grown around the RL rollouts to form the abstraction.
        self.rl_config = RLConfig(
            pi_arch=[64, 64],
            vf_arch=[64, 64],
            goal_reward=5,
            unsafe_penalty=-5,
            out_of_bounds_penalty=-5,
            per_step_cost=0.1,
            distance_cost=0.0,
            inflation_rate=[(-1, 1)],
        )
