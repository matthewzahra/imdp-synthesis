import numpy as np
from matplotlib import animation

from benchmarks.models import CartPoleDynamics
from core.box import Box
from core.rl.config import RLConfig


class CartPole(CartPoleDynamics):
    '''
    CartPole benchmark (cart position, cart velocity, pole angle, pole angular velocity).
    '''

    def __init__(self, args):
        CartPoleDynamics.__init__(self, args)

        # Plot the cart position against the pole angle
        self.plot_dimensions = [0, 2]

        self.state_space = Box([-2.4, -3.0, -0.25, -3.0], [2.4, 3.0, 0.25, 3.0], names=self.state_variables)
        self.state_grid = [200, 200, 200, 200]
        # Force on the cart
        self.input_space = Box([-10], [10])
        self.input_grid = [41]

        # Goal: balance the pole upright on the right side of the track
        self.goal = [self.state_space.restrict(position=(1, 2.4), velocity=(-1.5, 1.5), angle=(-0.05, 0.05),
                                               angular_velocity=(-1.0, 1.0))]
        self.critical = []

        # Start upright on the left side of the track
        self.x0 = np.array([-1, 0.0, 0, 0.0])

        # RL configuration: networks, PPO training, reward function, and the tube
        # grown around the RL rollouts to form the abstraction.
        self.rl_config = RLConfig(
            rl_algo="sac",
            total_timesteps= 5000000,
            goal_reward=20.0,
            unsafe_penalty=-20.0,
            out_of_bounds_penalty=-20.0,
            distance_cost=[0.1, 0.0, 0.0, 0.0],
            per_step_cost=0.05,
            inflation_rate=[(-7, 7), (-7, 7), (-7, 7), (-7, 7)],
            RL_actions_per_state=5,
        )

    def plot_simulation(self, args, stamp, partition, traces):
        print('Plot CartPole gif...')
        self.plot_trajectory_gif(np.array(traces[0]['x'])[:, [0, 2]],
                                 filename=str(args.output_dir / f'cartpole_{stamp}.gif'))

    def plot_trajectory_gif(self, trajectory, filename="cartpole_trajectory.gif"):
        """
        Plots a trajectory of the cartpole as an animation and stores it as a gif.

        Args:
            trajectory: np.ndarray of shape (T, 2), where each row is [cart_position, pole_angle]
            filename: Output filename for the gif
        """
        import matplotlib.pyplot as plt

        trajectory = np.asarray(trajectory, dtype=float)
        x_positions = trajectory[:, 0]
        thetas = trajectory[:, 1]

        x_lim = 2.4
        pole_len = 2 * self.length  # Physical length of the pole (self.length is half of it)
        cart_w, cart_h = 0.4, 0.2
        y_top = cart_h + pole_len + 0.3

        # Prepare figure
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_xlim(-x_lim - 1, x_lim + 1)
        ax.set_ylim(-0.5, y_top)
        ax.set_aspect('equal')
        ax.grid()

        # Track
        ax.axhline(0, color='black', lw=1)

        from matplotlib.patches import Rectangle

        # Obstacles hanging from the ceiling; their lower edge is measured from the pole's pivot
        for x_min, x_max, bottom in getattr(self, 'obstacles', []):
            ax.add_patch(Rectangle((x_min, cart_h + bottom), x_max - x_min, y_top - cart_h - bottom,
                                   facecolor='tab:red', edgecolor='darkred', hatch='//', alpha=0.6))

        cart = Rectangle((0, 0), cart_w, cart_h, facecolor='black')
        ax.add_patch(cart)
        pole, = ax.plot([], [], '-', lw=4, color='tab:brown')
        time_template = 'step {:d}'
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

        def init():
            cart.set_xy((-cart_w / 2, 0))
            pole.set_data([], [])
            time_text.set_text('')
            return cart, pole, time_text

        def animate(i):
            x = x_positions[i]
            theta = thetas[i]
            cart.set_xy((x - cart_w / 2, 0))
            # Pole pivots at the top-center of the cart; theta=0 is upright
            px = x + pole_len * np.sin(theta)
            py = cart_h + pole_len * np.cos(theta)
            pole.set_data([x, px], [cart_h, py])
            time_text.set_text(time_template.format(i))
            return cart, pole, time_text

        ani = animation.FuncAnimation(
            fig, animate, frames=len(trajectory), interval=50, blit=True, init_func=init
        )

        ani.save(filename, writer='pillow')
        plt.close(fig)
