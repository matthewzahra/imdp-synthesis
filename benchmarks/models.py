'''
Dynamics of the benchmarks. Each class defines the state variables, the noise, step(x, u, w), and step_set (bounds on
the noise-free successors of a box of states and inputs); the benchmarks in this package add their specification.
'''

from functools import partial
import logging

import jax
import jax.numpy as jnp
import numpy as np
import scipy

from benchmarks.base import Benchmark
from benchmarks.dynamics import setmath
from core.abstraction.partition import EPS

logger = logging.getLogger(__name__)


def wrap_theta(theta):
    return (theta + jnp.pi) % (2 * jnp.pi) - jnp.pi


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

    @partial(jax.jit, static_argnums=(0))
    def step_set(self, state_min, state_max, action_min, action_max):
        state_min, state_max = setmath.box(jnp.array(state_min), jnp.array(state_max))
        [x_min, y_min, theta_min] = state_min
        [x_max, y_max, theta_max] = state_max

        action_min, action_max = setmath.box(jnp.array(action_min), jnp.array(action_max))
        [u1_min, u2_min] = jnp.maximum(action_min, self.input_space.lower)
        [u1_max, u2_max] = jnp.minimum(action_max, self.input_space.upper)

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
        V_next = jnp.clip(V + self.beta * self.tau * u2 + noise[3], self.v_min + 1e-4, self.v_max - 1e-4)

        return jnp.array([x_next, y_next, theta_next, V_next])

    @partial(jax.jit, static_argnums=(0))
    def step_set(self, state_min, state_max, action_min, action_max):
        state_min, state_max = setmath.box(jnp.array(state_min), jnp.array(state_max))
        [x_min, y_min, theta_min, V_min] = state_min
        [x_max, y_max, theta_max, V_max] = state_max

        action_min, action_max = setmath.box(jnp.array(action_min), jnp.array(action_max))
        [u1_min, u2_min] = jnp.maximum(action_min, self.input_space.lower)
        [u1_max, u2_max] = jnp.minimum(action_max, self.input_space.upper)

        x_next = jnp.array([x_min, x_max]) + self.tau * jnp.concat(setmath.mult([V_min, V_max], setmath.cos(theta_min, theta_max)))
        y_next = jnp.array([y_min, y_max]) + self.tau * jnp.concat(setmath.mult([V_min, V_max], setmath.sin(theta_min, theta_max)))
        theta_next = jnp.array([theta_min, theta_max]) + self.tau * jnp.concat(setmath.mult([self.alpha_min, self.alpha_max], [u1_min, u1_max]))
        V_next = jnp.clip(jnp.array([V_min, V_max]) + self.beta * self.tau * jnp.array([u2_min, u2_max]), self.v_min + 1e-4, self.v_max - 1e-4)

        state_next = jnp.vstack((x_next, y_next, theta_next, V_next))

        return jnp.min(state_next, axis=1), jnp.max(state_next, axis=1)


class DroneDynamics(Benchmark):
    '''
    Drone moving in dim = 2 or 3 dimensions: a double integrator per dimension, with state (position, velocity) per
    dimension and an acceleration input per dimension. Velocities are damped quadratically (--damping) and saturate
    at [v_min, v_max].
    '''

    v_min = -2.0
    v_max = 2.0

    def __init__(self, args, dim=2):
        self._set_motion(args, dim)
        if dim == 2:
            self.set_noise(args,
                           gaussian=dict(variance=np.array([0.15, 0, 0.15, 0])**2, cells=[10, 1, 10, 1]),
                           triangular=dict(halfwidth=[0.15, 0, 0.15, 0], cells=[10, 1, 10, 1]))
        else:
            self.set_noise(args,
                           gaussian=dict(variance=np.array([0.05, 0, 0.05, 0, 0.05, 0])**2, cells=[5, 1, 5, 1, 5, 1]),
                           triangular=dict(halfwidth=[0.1, 0, 0.1, 0, 0.1, 0], cells=[10, 1, 10, 1, 10, 1]))

    def _set_motion(self, args, dim, agents=1, battery=False):
        '''
        Set the state variables and the matrices A, B of the motion: `agents` drones in `dim` dimensions each, plus a
        last, battery state (whose row of A is the identity) if battery.
        '''
        if dim not in (2, 3):
            raise ValueError(f'Drone dynamics only support dim in [2, 3], got {dim}')

        self.dim = dim
        self.tau = 1.0
        self.damping = args.damping

        variables =[f'{axis}_{quantity}' for axis in 'xyz'[:dim] for quantity in ('pos', 'vel')]
        if agents > 1:
            variables = [f'drone_{agent}_{variable}' for agent in range(1, agents + 1) for variable in variables]

        Ablock = np.array([[1, self.tau],
                           [0, 1]])
        Bblock = np.array([[self.tau**2/2],
                           [self.tau]])
        self.A = scipy.linalg.block_diag(*[Ablock] * (dim * agents))
        self.B = scipy.linalg.block_diag(*[Bblock] * (dim * agents))

        if battery:
            variables = variables + ['battery']
            self.A = scipy.linalg.block_diag(self.A, 1)
            self.B = np.vstack([self.B, np.zeros((1, self.B.shape[1]))])
        self.state_variables = variables

    def step(self, state, action, noise):
        state_next = jnp.dot(self.A, state) + jnp.dot(self.B, action) + noise
        state_next = state_next.at[1::2].add(-self.damping * state[1::2] * jnp.abs(state[1::2]))
        state_next = state_next.at[1::2].set(jnp.clip(state_next[1::2], self.v_min + 1e-4, self.v_max - 1e-4))

        return state_next

    @partial(jax.jit, static_argnums=(0))
    def step_set(self, state_min, state_max, action_min, action_max):
        return self._motion_set(state_min, state_max, action_min, action_max)

    def _motion_set(self, state_min, state_max, action_min, action_max):
        '''Bounds on the successors of the motion (the battery state, if any, stays constant).'''
        action_min = jnp.maximum(action_min, self.input_space.lower)
        action_max = jnp.minimum(action_max, self.input_space.upper)

        # Get vertices of the state and action boxes
        state_vertices = setmath.box2vertices(state_min, state_max)
        action_vertices = setmath.box2vertices(action_min, action_max)

        # Propagate dynamics for all vertices
        Ax = jnp.dot(self.A, state_vertices.T).T  # Shape (2^n, n)
        Ax = Ax.at[:, 1::2].add(-self.damping * state_vertices[:, 1::2] * jnp.abs(state_vertices[:, 1::2]))
        Bu = jnp.dot(self.B, action_vertices.T).T  # Shape (2^p, n)

        # Combine min/max to get the reachable set
        state_next_min = jnp.min(Ax, axis=0) + jnp.min(Bu, axis=0)
        state_next_max = jnp.max(Ax, axis=0) + jnp.max(Bu, axis=0)

        state_next_min = state_next_min.at[1::2].set(jnp.clip(state_next_min[1::2], self.v_min + 1e-4, self.v_max - 1e-4))
        state_next_max = state_next_max.at[1::2].set(jnp.clip(state_next_max[1::2], self.v_min + 1e-4, self.v_max - 1e-4))

        return state_next_min, state_next_max

    def plot_simulation(self, args, stamp, partition, traces):
        '''A drone moving in 3D: plot the traces in 3D, with Matplotlib and PyVista.'''
        if self.dim != 3:
            return
        from core.plotting.drone3d import plot_drone_3d_pyvista
        from core.plotting.traces import plot_traces_3d

        print('Plot Drone6D traces in 3D...')
        plot_traces_3d(args, stamp, [0, 2, 4], partition, self, traces, num_traces=100, filename="traces_3d")
        plot_drone_3d_pyvista(args, stamp, [0, 2, 4], partition, self, traces, num_traces=10)


class DroneDynamics_2agent(DroneDynamics):
    '''Two independent 2D drones: drone 1 in state dimensions 0-3 and input dimensions 0-1, drone 2 in 4-7 and 2-3.'''

    def __init__(self, args):
        self._set_motion(args, dim=2, agents=2)
        # Gaussian noise only acts on the x position of drone 1 and the y position of drone 2; the triangular noise is
        # that of a single 2D drone, for each drone
        self.set_noise(args,
                       gaussian=dict(variance=np.array([0.05, 0, 0, 0, 0, 0, 0.05, 0])**2,
                                     cells=[5, 1, 1, 1, 1, 1, 5, 1]),
                       triangular=dict(halfwidth=np.tile([0.15, 0, 0.15, 0], 2), cells=[10, 1, 10, 1] * 2))


class DroneDynamics_battery(DroneDynamics):
    '''
    DroneDynamics with a battery state (the last dimension): in the charging station, the battery charges by 10 per
    step up to max_charge; elsewhere it drains by 5 per step. Benchmarks define charging_station (a list of one Box).
    '''

    max_charge = 100

    def __init__(self, args, dim=2):
        self._set_motion(args, dim, battery=True)
        if dim == 2:
            self.set_noise(args,
                           gaussian=dict(variance=np.array([0.15, 0, 0.15, 0, 0])**2, cells=[10, 1, 10, 1, 1]),
                           triangular=dict(halfwidth=[0.1, 0, 0.1, 0, 0], cells=[10, 1, 10, 1, 1]))
        else:
            self.set_noise(args,
                           gaussian=dict(variance=np.array([0.05, 0, 0, 0, 0.05, 0, 0])**2,
                                         cells=[5, 1, 1, 1, 5, 1, 1]),
                           triangular=dict(halfwidth=[0.1, 0, 0.1, 0, 0.1, 0, 0], cells=[10, 1, 10, 1, 10, 1, 1]))

    def step(self, state, action, noise):
        state_next = super().step(state, action, noise)
        battery_idx = self.n - 1

        station = self.charging_station[0]
        in_station = jnp.all((state >= station.lower) & (state <= station.upper))
        new_battery = jnp.where(
            in_station,
            jnp.minimum(state_next[battery_idx] + 10, self.max_charge - 1e-4),
            state_next[battery_idx] - 5
        )
        return state_next.at[battery_idx].set(new_battery)

    @partial(jax.jit, static_argnums=(0))
    def step_set(self, state_min, state_max, action_min, action_max):
        state_next_min, state_next_max = self._motion_set(state_min, state_max, action_min, action_max)

        # Battery charging calculation (+10 when charging, capped at max_charge; -5 otherwise)
        cs_min = self.charging_station[0].lower
        cs_max = self.charging_station[0].upper

        # Same tolerances as the goal (containment) and critical (overlap) regions of the partition: a cell
        # that only touches the station does not intersect it, and cell bounds that round past a station face
        # (e.g. the top velocity cell's upper bound, 2.0000002 for v_max = 2) still count as inside
        entirely_inside = jnp.all((state_min >= cs_min - EPS / 2) & (state_max <= cs_max + EPS / 2))
        intersects = jnp.all((state_max >= cs_min + EPS) & (state_min <= cs_max - EPS))

        # entirely_inside: every state charges -> both bounds +10
        # partial overlap: some states charge, some drain -> min -5, max +10
        # no overlap: every state drains -> both bounds -5
        delta_min = jnp.where(entirely_inside, 10.0, -5.0)
        delta_max = jnp.where(intersects, 10.0, -5.0)

        battery_idx = self.n - 1
        new_min = jnp.minimum(state_next_min[battery_idx] + delta_min, self.max_charge - 1e-4)
        new_max = jnp.minimum(state_next_max[battery_idx] + delta_max, self.max_charge - 1e-4)
        state_next_min = state_next_min.at[battery_idx].set(new_min)
        state_next_max = state_next_max.at[battery_idx].set(new_max)

        return state_next_min, state_next_max


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

    @partial(jax.jit, static_argnums=(0))
    def step_set(self, state_min, state_max, action_min, action_max):
        state_min, state_max = setmath.box(jnp.array(state_min), jnp.array(state_max))
        [angle_min, velo_min] = state_min
        [angle_max, velo_max] = state_max

        action_min, action_max = setmath.box(jnp.array(action_min), jnp.array(action_max))
        u_min = jnp.maximum(action_min, self.input_space.lower)[0]
        u_max = jnp.minimum(action_max, self.input_space.upper)[0]

        velo_next = setmath.tuple2box((1 - self.b) * jnp.array([velo_min, velo_max])) + \
                    setmath.tuple2box(self.tau * 3 * self.G / (2 * self.l) * setmath.tuple2box(setmath.sin(angle_min, angle_max))) + \
                    setmath.tuple2box(self.tau * 3.0 / (self.m * self.l ** 2) * jnp.array([u_min, u_max]))

        angle_next = jnp.array([angle_min, angle_max]) + self.tau * velo_next

        state_next = jnp.vstack((angle_next, velo_next))

        return jnp.min(state_next, axis=1), jnp.max(state_next, axis=1)


class MountainCarDynamics(Benchmark):
    '''Mountain car (as in Gymnasium, with a continuous force input) with state (position, velocity).'''

    def __init__(self, args):
        self.tau = 1
        self.state_variables = ['position', 'velocity']

        # Parameters
        self.max_speed = 0.07
        self.gravity = 0.0025
        self.power = 0.0015

        self.set_noise(args,
                       gaussian=dict(variance=np.array([0.005, 0.0005])**2, cells=[10, 10]),
                       triangular=dict(halfwidth=[0.005, 0.0005], cells=[10, 10]))

    def step(self, state, action, noise):

        position, velocity = state[0], state[1]

        velocity = velocity + self.tau * (action[0] * self.power - self.gravity * jnp.cos(3 * position))
        velocity = jnp.clip(velocity, -self.max_speed+1e-4, self.max_speed-1e-4)
        position = position + self.tau * velocity + noise[0]
        velocity = velocity + noise[1]

        return jnp.array([position, velocity])

    @partial(jax.jit, static_argnums=(0))
    def step_set(self, state_min, state_max, action_min, action_max):
        state_min, state_max = setmath.box(jnp.array(state_min), jnp.array(state_max))
        [pos_min, velo_min] = state_min
        [pos_max, velo_max] = state_max

        action_min, action_max = setmath.box(jnp.array(action_min), jnp.array(action_max))
        u_min = jnp.maximum(action_min, self.input_space.lower)[0]
        u_max = jnp.minimum(action_max, self.input_space.upper)[0]

        velo_next = jnp.array([velo_min, velo_max]) + \
                    setmath.tuple2box(self.tau * -self.gravity * setmath.tuple2box(setmath.cos(3 * pos_min, 3 * pos_max)) ) + \
                    setmath.tuple2box(self.tau * self.power * jnp.array([u_min, u_max]))

        # Match the real dynamics (see step): velocity saturates at the speed limit before it
        # drives the position update. The clip is monotone, so applying it to the [lb, ub]
        # interval is exact. Without it, high-velocity cells (the goal-approach corridor) overshoot
        # the velocity grid boundary (±max_speed) and get mapped to the absorbing/failure state.
        velo_next = jnp.clip(velo_next, -self.max_speed + 1e-4, self.max_speed - 1e-4)

        pos_next = jnp.array([pos_min, pos_max]) + self.tau * velo_next

        state_next = jnp.vstack((pos_next, velo_next))

        return jnp.min(state_next, axis=1), jnp.max(state_next, axis=1)


class CartPoleDynamics(Benchmark):
    '''
    CartPole dynamics model, based on the Gymnasium CartPole environment.
    The state consists of the cart position, cart velocity, pole angle, and pole angular velocity.
    The action is a force applied to the cart. The dynamics are nonlinear and include process noise.

    Differences with the original CartPole environment:
    - The time step is 0.02 * 5 seconds instead of 0.02 seconds.
    - Original environment has no process noise.
    - Original environment has a discrete action space (left/right), while this model uses a continuous action space (force).
    '''

    def __init__(self, args):
        self.tau = 0.02 * 5
        self.state_variables = ['position', 'velocity', 'angle', 'angular_velocity']

        # CartPole parameters (Gymnasium CartPole convention)
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # Actually half the pole's length
        self.polemass_length = self.masspole * self.length

        # Gaussian noise acts on the cart velocity, triangular noise on the cart position and the pole angle
        self.set_noise(args,
                       gaussian=dict(variance=np.array([0, 0.001, 0, 0])**2, cells=[1, 25, 1, 1]),
                       triangular=dict(halfwidth=[0.005, 0, 0.005, 0], cells=[10, 1, 10, 1]))

    def step(self, state, action, noise):
        x, x_dot, theta, theta_dot = state[0], state[1], state[2], state[3]
        force = action[0]

        costheta = jnp.cos(theta)
        sintheta = jnp.sin(theta)

        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # Euler integration
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        return jnp.array([x, x_dot, theta, theta_dot]) + noise

    @partial(jax.jit, static_argnums=(0))
    def step_set(self, state_min, state_max, action_min, action_max):
        state_min, state_max = setmath.box(jnp.array(state_min), jnp.array(state_max))
        [x_min, xd_min, th_min, thd_min] = state_min
        [x_max, xd_max, th_max, thd_max] = state_max

        action_min, action_max = setmath.box(jnp.array(action_min), jnp.array(action_max))
        F_min = jnp.maximum(action_min, self.input_space.lower)[0]
        F_max = jnp.minimum(action_max, self.input_space.upper)[0]

        # Interval bounds on the trigonometric terms of the pole angle
        s_min, s_max = setmath.sin(th_min, th_max)
        c_min, c_max = setmath.cos(th_min, th_max)

        # theta_dot^2
        thd2_min, thd2_max = setmath.square(thd_min, thd_max)

        # temp = (F + polemass_length * theta_dot^2 * sin(theta)) / total_mass
        # (setmath.mult/div return shape-(1,) arrays, so we squeeze back to scalars)
        q_min, q_max = setmath.mult((thd2_min, thd2_max), (s_min, s_max))
        q_min, q_max = q_min[0], q_max[0]
        temp_min = (F_min + self.polemass_length * q_min) / self.total_mass
        temp_max = (F_max + self.polemass_length * q_max) / self.total_mass

        # thetaacc = (g*sin - cos*temp) / (length * (4/3 - masspole*cos^2/total_mass))
        b_min, b_max = setmath.mult((c_min, c_max), (temp_min, temp_max))
        b_min, b_max = b_min[0], b_max[0]
        num_min = self.gravity * s_min - b_max
        num_max = self.gravity * s_max - b_min

        c2_min, c2_max = setmath.square(c_min, c_max)
        k = self.masspole / self.total_mass
        # Denominator is guaranteed positive (4/3 - k*cos^2 >= 4/3 - k > 0)
        denom_min = self.length * (4.0 / 3.0 - k * c2_max)
        denom_max = self.length * (4.0 / 3.0 - k * c2_min)

        thacc_min, thacc_max = setmath.div((num_min, num_max), (denom_min, denom_max))
        thacc_min, thacc_max = thacc_min[0], thacc_max[0]

        # xacc = temp - polemass_length * thetaacc * cos / total_mass
        e_min, e_max = setmath.mult((thacc_min, thacc_max), (c_min, c_max))
        e_min, e_max = e_min[0], e_max[0]
        coef = self.polemass_length / self.total_mass
        xacc_min = temp_min - coef * e_max
        xacc_max = temp_max - coef * e_min

        # Euler integration
        x_next = jnp.array([x_min + self.tau * xd_min, x_max + self.tau * xd_max])
        xd_next = jnp.array([xd_min + self.tau * xacc_min, xd_max + self.tau * xacc_max])
        th_next = jnp.array([th_min + self.tau * thd_min, th_max + self.tau * thd_max])
        thd_next = jnp.array([thd_min + self.tau * thacc_min, thd_max + self.tau * thacc_max])

        state_next = jnp.vstack((x_next, xd_next, th_next, thd_next))

        return jnp.min(state_next, axis=1), jnp.max(state_next, axis=1)


class AffineDynamics(Benchmark):
    '''
    Affine dynamics with additive noise: x' = A x + B u + q + w.

    Benchmarks call AffineDynamics.__init__ with their matrices and noise (see Benchmark.set_noise) and then define
    their specification.
    '''

    def __init__(self, args, A, B, q, gaussian=None, triangular=None, state_variables=None):
        '''
        :param args: Argument namespace (provides noise_distr)
        :param A: State matrix, shape [n, n]
        :param B: Input matrix, shape [n, p]
        :param q: Constant drift, shape [n]
        :param gaussian: Gaussian noise, dict(variance=..., cells=...); None if not supported
        :param triangular: Triangular noise, dict(halfwidth=..., cells=...); None if not supported
        :param state_variables: Names of the state variables (default x1, ..., xn)
        '''
        self.A = np.asarray(A, dtype=float)
        self.B = np.asarray(B, dtype=float)
        self.q = np.asarray(q, dtype=float).reshape(-1)
        self.tau = 1
        n = self.A.shape[0]
        self.state_variables = list(state_variables) if state_variables is not None else [f'x{i + 1}' for i in range(n)]
        self.set_noise(args, gaussian=gaussian, triangular=triangular)

    def step(self, state, action, noise):
        return jnp.dot(self.A, state) + jnp.dot(self.B, action) + self.q + noise

    @partial(jax.jit, static_argnums=(0))
    def step_set(self, state_min, state_max, action_min, action_max):
        action_min = jnp.maximum(action_min, self.input_space.lower)
        action_max = jnp.minimum(action_max, self.input_space.upper)

        # The image of a box under an affine map is bounded exactly by its center +- |M| times its radius
        state_center, state_radius = (state_min + state_max) / 2, (state_max - state_min) / 2
        action_center, action_radius = (action_min + action_max) / 2, (action_max - action_min) / 2

        center = jnp.dot(self.A, state_center) + jnp.dot(self.B, action_center) + self.q
        radius = jnp.dot(np.abs(self.A), state_radius) + jnp.dot(np.abs(self.B), action_radius)

        return center - radius, center + radius
