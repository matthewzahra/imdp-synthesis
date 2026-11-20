import numpy as np
from benchmarks.models import AffineDynamics
from core.rl.config import RLConfig


class DoubleIntegrator(AffineDynamics):
    '''
    Double integrator benchmark, with a 2D state space (position, velocity) and a 1D control input space.
    '''

    def __init__(self, args):
        tau = 1.0
        A = np.array([[1, tau],
                      [0, 1]])
        B = np.array([[tau**2 / 2],
                      [tau]])
        
        AffineDynamics.__init__(self, args, A=A, B=B, q=[0, 0],
                                noise_var=np.array([0.15, 0.15])**2, noise_halfwidth=[0.15, 0.15], noise_cells=[10, 10],
                                state_variables=['position', 'velocity'], wrap=[False, False])

        self.plot_dimensions = [0, 1]

        self.set_spec()

    def set_spec(self):
        '''
        Set the abstraction parameters and the reach-avoid specification.
        '''
        
        self.partition = {}

        # Authority limit for the control u, both positive and negative
        self.uMin = [-5]
        self.uMax = [5]
        self.num_actions = [21]

        self.partition['boundary'] = np.array([[-21, -10.5], [21, 10.5]])
        self.partition['number_per_dim'] = np.array([21, 21])

        self.goal = np.array([
            [[-4, -2], [4, 2]]
        ], dtype=float)

        self.critical = np.array([
        ], dtype=float)

        self.x0 = np.array([0, -8])

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
            inflation_rate=[(-2, 2), (-1, 1)],
        )

        return
