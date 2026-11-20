import numpy as np

from benchmarks.models import DroneDynamics, DroneDynamics_battery
from core.box import Box
from core.rl.config import RLConfig


# Obstacles of Drone6D and Drone6D_battery, as position bounds
DRONE6D_OBSTACLES = [
    # Hole 1
    dict(x_pos=(-11, -5), y_pos=(-1, 9), z_pos=(-7, -5)),
    dict(x_pos=(-11, -5), y_pos=(5, 9), z_pos=(-5, 5)),
    dict(x_pos=(-11, -5), y_pos=(-1, 3), z_pos=(-5, 3)),
    # Hole 2
    dict(x_pos=(-1, 3), y_pos=(1, 9), z_pos=(-7, -1)),
    dict(x_pos=(-1, 3), y_pos=(1, 9), z_pos=(3, 5)),
    dict(x_pos=(-1, 3), y_pos=(1, 3), z_pos=(-1, 3)),
    dict(x_pos=(-1, 3), y_pos=(7, 9), z_pos=(-1, 3)),
    # Tower
    dict(x_pos=(-1, 3), y_pos=(-3, 1), z_pos=(-7, 7)),
    # Wall between routes
    dict(x_pos=(3, 9), y_pos=(-3, 1), z_pos=(-7, -1)),
    # Long route obstacles
    dict(x_pos=(-1, 3), y_pos=(-9, -3), z_pos=(-7, -5)),
    # Overhanging
    dict(x_pos=(-1, 3), y_pos=(-9, -3), z_pos=(3, 7)),
    # Small last obstacle
    dict(x_pos=(11, 15), y_pos=(-9, -5), z_pos=(-7, -5)),
    # Obstacle next to goal
    dict(x_pos=(9, 15), y_pos=(5, 9), z_pos=(-7, 1)),
]


class Drone6D(DroneDynamics):
    '''
    Drone benchmark, with a 6D state space and a 3D control input space.
    '''

    def __init__(self, args):
        DroneDynamics.__init__(self, args, dim=3)

        self.plot_dimensions = [0, 2]

        v_min, v_max = self.v_min, self.v_max
        self.state_space = Box([-17, v_min, -9, v_min, -7, v_min], [17, v_max, 9, v_max, 7, v_max],
                               names=self.state_variables)
        self.state_grid = [68, 8, 36, 8, 28, 8]
        self.input_space = Box([-1, -1, -1], [1, 1, 1])
        self.input_grid = [5, 5, 5]

        region = self.state_space.restrict
        self.goal = [region(x_pos=(11, 15), y_pos=(1, 5), z_pos=(-7, -3))]
        self.critical = [region(**obstacle) for obstacle in DRONE6D_OBSTACLES]

        self.x0 = np.array([-14.5, 0.01, 6, 0.01, 2, 0.01])

        # RL configuration: networks, PPO training, reward function, and the tube
        # grown around the RL rollouts to form the abstraction.
        self.rl_config = RLConfig(
            rl_algo="sac",
            total_timesteps=10000000,
            RL_actions_per_state=27,
            proximity_dims=[0, 2, 4],
            proximity_penalty=0.5,
            per_step_cost=0.05,
            inflation_rate=[(-3, 3), (-2, 2), (-3, 3), (-2, 2), (-3, 3), (-2, 2)],
        )


class Drone6D_small(DroneDynamics):
    '''
    Reduced Drone6D configuration for faster debugging, with a 6D state space and a 3D control input space.
    '''

    def __init__(self, args):
        DroneDynamics.__init__(self, args, dim=3)

        self.plot_dimensions = [0, 2]

        v_min, v_max = self.v_min, self.v_max
        self.state_space = Box([-7, v_min, -7, v_min, -7, v_min], [7, v_max, 7, v_max, 7, v_max],
                               names=self.state_variables)
        self.state_grid = [28, 10, 28, 10, 28, 10]
        self.input_space = Box([-1, -1, -1], [1, 1, 1])
        self.input_grid = [5, 5, 5]

        region = self.state_space.restrict
        self.goal = [region(x_pos=(3, 7), y_pos=(3, 7))]
        self.critical = [
            region(x_pos=(-7, -1), y_pos=(1, 3)),
            region(x_pos=(3, 7), y_pos=(-7, -3)),
        ]

        self.x0 = np.array([-5.5, 0.01, -5.5, 0.01, 0.01, 0.01])

        # RL configuration: networks, PPO training, reward function, and the tube
        # grown around the RL rollouts to form the abstraction.
        self.rl_config = RLConfig(
            pi_arch=[32, 32],
            vf_arch=[32, 32],
            total_timesteps=100000,
            eval_episodes=1000,
            goal_reward=5,
            unsafe_penalty=-5,
            out_of_bounds_penalty=-5,
            per_step_cost=0.1,
            distance_cost=0.0,
            RL_actions_per_state=27,
            inflation_rate=[(-2, 2), (-1, 1), (-2, 2), (-1, 1), (-2, 2), (-1, 1)],
        )


class Drone6D_battery(DroneDynamics_battery):
    '''
    Drone benchmark with 6D motion, one battery state, and 3D control.
    '''

    def __init__(self, args):
        DroneDynamics_battery.__init__(self, args, dim=3)

        self.plot_dimensions = [0, 2]

        # Same layout as Drone6D (bounds, obstacles, goal, and initial state), plus the battery dimension
        v_min, v_max = self.v_min, self.v_max
        self.state_space = Box([-17, v_min, -9, v_min, -7, v_min, 0], [17, v_max, 9, v_max, 7, v_max, self.max_charge],
                               names=self.state_variables)
        self.state_grid = [68, 8, 36, 8, 28, 8, 20]
        self.input_space = Box([-1, -1, -1], [1, 1, 1])
        self.input_grid = [5, 5, 5]

        region = self.state_space.restrict
        self.goal = [region(x_pos=(11, 15), y_pos=(1, 5), z_pos=(-7, -3), battery=(20, self.max_charge))]
        self.critical = [region(**obstacle) for obstacle in DRONE6D_OBSTACLES]

        # Charging pad in the minimal (x, y, z) corner of the state space
        self.charging_station = [region(x_pos=(-15, -7), y_pos=(-9, -4), z_pos=(-7, 0))]

        self.x0 = np.array([-14.5, 0.01, 6, 0.01, 2, 0.01, 50])

        # RL configuration: networks, PPO training, reward function, and the tube
        # grown around the RL rollouts to form the abstraction.
        self.rl_config = RLConfig(
            rl_algo="sac",
            total_timesteps=25000000,
            RL_actions_per_state=27,
            proximity_dims=[0, 2, 4],
            proximity_penalty=0.5,
            per_step_cost=0.05,
            unsafe_penalty=-10,
            out_of_bounds_penalty=-10,
            inflation_rate=[(-3, 3), (-2, 2), (-3, 3), (-2, 2), (-3, 3), (-2, 2), (-2, 2)],
        )
