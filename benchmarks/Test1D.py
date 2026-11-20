import numpy as np
from benchmarks.models import AffineDynamics
from core.rl.config import RLConfig


class Test1D(AffineDynamics):
    '''
    Simple 1D test benchmark (x' = x + u + w), with a 1D state space and a 1D control input space.
    '''

    def __init__(self, args):
        AffineDynamics.__init__(self, args, A=[[1]], B=[[1]], q=[0],
                                noise_var=np.array([0.2])**2, noise_halfwidth=[0.2], noise_cells=[10],
                                state_variables=['position'], wrap=[False])

        self.plot_dimensions = [0]

        self.set_spec()

    def set_spec(self):
        '''
        Set the abstraction parameters and the reach-avoid specification.
        '''
        
        self.partition = {}

        # Authority limit for the control u, both positive and negative
        self.uMin = [-1]
        self.uMax = [1]
        self.num_actions = [3]

        self.partition['boundary'] = np.array([[-5], [5]])
        self.partition['number_per_dim'] = np.array([10])

        self.goal = np.array([
            [[-1], [1]]
        ], dtype=float)

        self.critical = np.array([
        ], dtype=float)

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

        return
