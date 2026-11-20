import numpy as np

from benchmarks.models import AffineDynamics
from core.box import Box
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
                                gaussian=dict(variance=np.array([0.15, 0.15])**2, cells=[10, 10]),
                                triangular=dict(halfwidth=[0.15, 0.15], cells=[10, 10]),
                                state_variables=['position', 'velocity'])

        self.plot_dimensions = [0, 1]

        self.state_space = Box([-21, -10.5], [21, 10.5], names=self.state_variables)
        self.state_grid = [21, 21]
        self.input_space = Box([-5], [5])
        self.input_grid = [21]

        self.goal = [self.state_space.restrict(position=(-4, 4), velocity=(-2, 2))]
        self.critical = []

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
