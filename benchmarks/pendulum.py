'''
Inverted pendulum benchmark (as in Gymnasium): swing up the pendulum and balance it upright.
'''

import jax.numpy as jnp
from matplotlib import animation
import numpy as np

from benchmarks.base import Benchmark, wrap_theta
from benchmarks.dynamics import setmath
from core.box import Box
from core.rl.config import RLConfig


class PendulumDynamics(Benchmark):
    '''Inverted pendulum (as in Gymnasium) with state (angle, angular velocity) and a torque input.'''

    wrapped = ('angle',)

    def __init__(self, args):
        self.tau = 0.05
        self.state_variables = ['angle', 'velocity']

        # Pendulum parameters
        self.G = 10
        self.m = 1.0
        self.l = 1.0
        self.b = 0.0 # Gymnasium pendulum does not have damping

        self.set_noise(args,
                       gaussian=dict(variance=np.array([0.03, 0.1])**2, cells=[25, 25]),
                       triangular=dict(halfwidth=[0.03, 0.1], cells=[10, 10]))

    def step(self, state, action, noise):

        new_velo = (1 - self.b) * state[1] + \
                   (3 * self.G / (2 * self.l) * jnp.sin(state[0])) * self.tau + \
                   (3.0 / (self.m * self.l**2) * action[0]) * self.tau
        new_angle = wrap_theta(state[0] + self.tau * new_velo + noise[0])
        new_velo = new_velo + noise[1]

        return jnp.array([new_angle, new_velo])

    def successor_bounds(self, state_min, state_max, action_min, action_max):
        state_min, state_max = setmath.box(jnp.array(state_min), jnp.array(state_max))
        [angle_min, velo_min] = state_min
        [angle_max, velo_max] = state_max

        action_min, action_max = setmath.box(jnp.array(action_min), jnp.array(action_max))
        u_min = action_min[0]
        u_max = action_max[0]

        velo_next = setmath.tuple2box((1 - self.b) * jnp.array([velo_min, velo_max])) + \
                    setmath.tuple2box(self.tau * 3 * self.G / (2 * self.l) * setmath.tuple2box(setmath.sin(angle_min, angle_max))) + \
                    setmath.tuple2box(self.tau * 3.0 / (self.m * self.l ** 2) * jnp.array([u_min, u_max]))

        angle_next = jnp.array([angle_min, angle_max]) + self.tau * velo_next

        state_next = jnp.vstack((angle_next, velo_next))

        return jnp.min(state_next, axis=1), jnp.max(state_next, axis=1)


class Pendulum(PendulumDynamics):
    '''
    Pendulum benchmark.
    '''

    def __init__(self, args):
        PendulumDynamics.__init__(self, args)

        self.plot_dimensions = [0, 1]

        self.state_space = Box([-np.pi, -8], [np.pi, 8], names=self.state_variables)
        self.state_grid = [300, 300]
        self.input_space = Box([-2], [2])
        self.input_grid = [11]

        self.goal = [self.state_space.restrict(angle=(-0.1*np.pi, 0.1*np.pi), velocity=(-1, 1))]
        self.critical = []

        self.x0 = np.array([0.99*np.pi, 0])

        # RL configuration: networks, PPO training, reward function, and the tube
        # grown around the RL rollouts to form the abstraction.
        self.rl_config = RLConfig(
            rl_algo='ppo',
            total_timesteps=1000000,
            per_step_cost=0.05,
            inflation_rate=[(-5, 5), (-5, 5)],
            RL_actions_per_state=5,
        )

    def plot_simulation(self, args, stamp, partition, traces):
        print('Plot Pendulum gif...')
        self.plot_trajectory_gif(np.array(traces[0]['x'])[:, 0],
                                 filename=str(args.output_dir / f'pendulum_{stamp}.gif'))

    def plot_trajectory_gif(self, trajectory, filename="pendulum_trajectory.gif"):
        """
        Plots a trajectory of the pendulum as an animation and stores it as a gif.

        Args:
            trajectory: np.ndarray of shape (T, 2), where each row is [theta, theta_dot]
            filename: Output filename for the gif
        """
        import matplotlib.pyplot as plt

        # Prepare figure
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.grid()

        line, = ax.plot([], [], 'o-', lw=5)
        time_template = 'step {:d}'
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

        L = 1.0  # Assume pendulum length is 1

        # fig.tight_layout()

        def init():
            line.set_data([], [])
            time_text.set_text('')
            return line, time_text

        def animate(i):
            theta = trajectory[i]
            x = L * np.sin(theta)
            y = L * np.cos(theta)  # Flip the sign to invert the y-axis
            line.set_data([0, x], [0, y])
            time_text.set_text(time_template.format(i))
            return line, time_text

        ani = animation.FuncAnimation(
            fig, animate, frames=len(trajectory), interval=50, blit=True, init_func=init
        )

        ani.save(filename, writer='pillow')
        plt.close(fig)
