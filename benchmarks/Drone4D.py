import numpy as np

from benchmarks.models import DroneDynamics, DroneDynamics_battery
from core.box import Box
from core.rl.config import RLConfig


class Drone4D(DroneDynamics):
    '''
    Drone benchmark, with a 4D state space and a 2D control input space.
    '''

    def __init__(self, args):
        DroneDynamics.__init__(self, args, dim=2)

        self.plot_dimensions = [0, 2]

        v_min, v_max = self.v_min, self.v_max
        self.state_space = Box([-20.0, v_min, -20.0, v_min], [20.0, v_max, 20.0, v_max], names=self.state_variables)
        self.state_grid = [80, 10, 80, 10]
        self.input_space = Box([-1, -1], [1, 1])
        self.input_grid = [5, 5]

        region = self.state_space.restrict
        self.goal = [region(x_pos=(12.0, 18.0), y_pos=(12.0, 18.0))]
        self.critical = [
            region(x_pos=(-20.0, 8.0), y_pos=(-7.0, -4.0)),     # Lower wall with right-side opening
            region(x_pos=(-8.0, 20.0), y_pos=(4.0, 7.0)),       # Upper wall with left-side opening
            region(x_pos=(-4.0, 4.0), y_pos=(-14.0, -10.0)),    # Lower lane obstacle
            region(x_pos=(-4.0, 4.0), y_pos=(10.0, 14.0)),      # Upper lane obstacle
        ]

        self.x0 = np.array([-15.0, 0.01, -15.0, 0.01])

        # RL configuration: networks, PPO training, reward function, and the tube
        # grown around the RL rollouts to form the abstraction.
        self.rl_config = RLConfig(
            rl_algo="ppo",
            # TODO: Long training is still needed here; can we reduce that?
            total_timesteps=3000000,
            RL_actions_per_state=9,
            inflation_rate=[(-4, 4), (-2, 2), (-4, 4), (-2, 2)],
            proximity_dims = [0, 2],
            proximity_penalty=0.5,
            per_step_cost=0.05,
        )


class Drone4D_battery(DroneDynamics_battery):
    '''
    Drone benchmark, with a 4D state space and a 2D control input space, plus a battery state.
    '''

    def __init__(self, args):
        DroneDynamics_battery.__init__(self, args, dim=2)

        self.plot_dimensions = [0, 2]

        v_min, v_max = self.v_min, self.v_max
        self.state_space = Box([-10, v_min, -10, v_min, 0], [10, v_max, 10, v_max, self.max_charge],
                               names=self.state_variables)
        self.state_grid = [40, 10, 40, 10, 20]
        self.input_space = Box([-1, -1], [1, 1])
        self.input_grid = [5, 5]

        region = self.state_space.restrict
        self.goal = [region(x_pos=(6, 10), y_pos=(6, 10), battery=(50, self.max_charge))]
        self.critical = [
            # Obstacle in the center
            region(x_pos=(-5, 5), y_pos=(-1, 1)),
            region(x_pos=(-1, 1), y_pos=(-5, 5)),
        ]
        self.charging_station = [region(x_pos=(-9, -6), y_pos=(-5, 5))]

        self.x0 = np.array([0.01, 0.01, -9, 0.01, 50])

        # RL configuration: networks, PPO training, reward function, and the tube
        # grown around the RL rollouts to form the abstraction.
        self.rl_config = RLConfig(
            rl_algo="ppo",
            total_timesteps=1000000,
            proximity_dims = [0, 2],
            proximity_penalty=0.5,
            per_step_cost=0.05,
            inflation_rate=[(-4, 4), (-2, 2), (-4, 4), (-2, 2), (-2, 2)],
            RL_actions_per_state=25,
        )
