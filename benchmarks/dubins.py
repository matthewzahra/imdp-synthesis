'''
Dubins vehicle benchmarks: a 3D vehicle with a speed input (Dubins3D), and a 4D vehicle with velocity as a state
(Dubins4D).
'''

import logging

import jax.numpy as jnp
import numpy as np

from benchmarks.base import Benchmark, wrap_theta
from benchmarks.dynamics import setmath
from core.box import Box
from core.rl.config import RLConfig

logger = logging.getLogger(__name__)


class DubinsDynamics3D(Benchmark):
    '''Dubins vehicle with state (x, y, angle) and input (turn rate, speed); noise on the angle only.'''

    wrapped = ('angle',)

    def __init__(self, args):
        self.tau = 1
        self.state_variables = ['x', 'y', 'angle']

        self.alpha_min = 0.85
        self.alpha_max = 0.85
        self.alpha = 0.85

        self.set_noise(args,
                       gaussian=dict(variance=np.array([0, 0, 0.25])**2, cells=[1, 1, 10]),
                       triangular=dict(halfwidth=[0, 0, 0.2], cells=[1, 1, 10]))

    def step(self, state, action, noise):
        x, y, theta = state[0], state[1], state[2]
        u1, u2 = action[0], action[1]
        x_next = x + self.tau * u2 * jnp.cos(theta)
        y_next = y + self.tau * u2 * jnp.sin(theta)
        theta_next = wrap_theta(theta + self.tau * self.alpha * u1 + noise[2])

        return jnp.array([x_next, y_next, theta_next])

    def successor_bounds(self, state_min, state_max, action_min, action_max):
        state_min, state_max = setmath.box(jnp.array(state_min), jnp.array(state_max))
        [x_min, y_min, theta_min] = state_min
        [x_max, y_max, theta_max] = state_max

        action_min, action_max = setmath.box(jnp.array(action_min), jnp.array(action_max))
        [u1_min, u2_min] = action_min
        [u1_max, u2_max] = action_max

        x_next = jnp.array([x_min, x_max]) + self.tau * jnp.concat(setmath.mult([u2_min, u2_max], setmath.cos(theta_min, theta_max)))
        y_next = jnp.array([y_min, y_max]) + self.tau * jnp.concat(setmath.mult([u2_min, u2_max], setmath.sin(theta_min, theta_max)))
        theta_next = jnp.array([theta_min, theta_max]) + self.tau * jnp.concat(setmath.mult([self.alpha_min, self.alpha_max], [u1_min, u1_max]))

        state_next = jnp.vstack((x_next, y_next, theta_next))

        return jnp.min(state_next, axis=1), jnp.max(state_next, axis=1)


class DubinsDynamics4D(Benchmark):
    '''
    Dubins vehicle with state (x, y, angle, velocity) and input (turn rate, acceleration). With --model_version 1 or 2,
    the turn-rate gain alpha is uncertain within an interval.
    '''

    wrapped = ('angle',)

    def __init__(self, args):
        self.tau = 1
        self.state_variables = ['x', 'y', 'angle', 'velocity']

        self.v_min = -0.5
        self.v_max = 0.5

        if args.model_version == 0:
            logger.info('- Load Dubins without parameter uncertainty')
            self.alpha_min, self.alpha_max = 0.85, 0.85
        elif args.model_version == 1:
            logger.info('- Load Dubins with uncertain parameters in the interval [0.80,0.90]')
            self.alpha_min, self.alpha_max = 0.80, 0.90
        else:
            logger.info('- Load Dubins with uncertain parameters in the interval [0.75,0.95]')
            self.alpha_min, self.alpha_max = 0.75, 0.95
        self.alpha = 0.85
        self.beta = 0.2

        self.set_noise(args,
                       gaussian=dict(variance=np.array([0.0, 0.0, 0.1, 0.0])**2, cells=[1, 1, 25, 1]),
                       triangular=dict(halfwidth=[0, 0, 0.1, 0], cells=[1, 1, 10, 1]))

    def step(self, state, action, noise):
        x, y, theta, V = state[0], state[1], state[2], state[3]
        u1, u2 = action[0], action[1]
        x_next = x + self.tau * V * jnp.cos(theta) + noise[0]
        y_next = y + self.tau * V * jnp.sin(theta) + noise[1]
        theta_next = wrap_theta(theta + self.tau * self.alpha * u1 + noise[2])
        V_next = self.saturate(V + self.beta * self.tau * u2 + noise[3], self.v_min, self.v_max)

        return jnp.array([x_next, y_next, theta_next, V_next])

    def successor_bounds(self, state_min, state_max, action_min, action_max):
        state_min, state_max = setmath.box(jnp.array(state_min), jnp.array(state_max))
        [x_min, y_min, theta_min, V_min] = state_min
        [x_max, y_max, theta_max, V_max] = state_max

        action_min, action_max = setmath.box(jnp.array(action_min), jnp.array(action_max))
        [u1_min, u2_min] = action_min
        [u1_max, u2_max] = action_max

        x_next = jnp.array([x_min, x_max]) + self.tau * jnp.concat(setmath.mult([V_min, V_max], setmath.cos(theta_min, theta_max)))
        y_next = jnp.array([y_min, y_max]) + self.tau * jnp.concat(setmath.mult([V_min, V_max], setmath.sin(theta_min, theta_max)))
        theta_next = jnp.array([theta_min, theta_max]) + self.tau * jnp.concat(setmath.mult([self.alpha_min, self.alpha_max], [u1_min, u1_max]))
        V_next = self.saturate(jnp.array([V_min, V_max]) + self.beta * self.tau * jnp.array([u2_min, u2_max]), self.v_min, self.v_max)

        state_next = jnp.vstack((x_next, y_next, theta_next, V_next))

        return jnp.min(state_next, axis=1), jnp.max(state_next, axis=1)


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
