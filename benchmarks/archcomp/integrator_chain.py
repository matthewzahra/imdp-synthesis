'''
Integrator chain (IC) benchmark of the ARCH-COMP stochastic models category (ARCH-COMP20, Sec. 4.2).

For a chain of n integrators with sampling time Ns = 0.1 (0-indexed):

    x_i[k+1] = x_i[k] + sum_{j > i} Ns^(j-i) / (j-i)! x_j[k] + Ns^(n-i) / (n-i)! u[k] + w_i[k],   w ~ N(0, 0.01 I).

The noise variance 0.01 is that of ArchCompStochasticModels.jl (integrator_chain) and IMPaCT's ARCH-COMP25 code.
The ARCH-COMP25 report states variance 0.1 for PRoTECT's variant, whose code uses 0.01.
'''

from math import factorial

import numpy as np

from benchmarks.archcomp.common import no_control
from benchmarks.models import AffineDynamics
from core.box import Box
from core.rl.config import RLConfig


def integrator_chain_matrices(n, Ns=0.1):
    '''State matrix A [n, n] and input matrix B [n, 1] of a chain of n integrators.'''
    A = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            A[i, j] = Ns ** (j - i) / factorial(j - i)
    B = np.array([[Ns ** (n - i) / factorial(n - i)] for i in range(n)])
    return A, B


class IntegratorChain2D(AffineDynamics):
    '''
    IMPaCT's ARCH-COMP25 variant: reach the target [-8, 8]^2 within 5 steps, without leaving [-10, 10]^2.

    Input u in [-1, 1]. Grid as in IMPaCT's entry: cells of width 0.5, five inputs spaced 0.5 apart. The benchmark
    defines no initial state; x0 lies outside the target, which is reachable from it within 5 steps.
    Sources: ARCH-COMP25 report (arXiv:2506.17602), Sec. "Benchmarking IMPaCT"; IMPaCT's IC_reach_2d.cpp.
    '''

    def __init__(self, args):
        A, B = integrator_chain_matrices(2)
        AffineDynamics.__init__(self, args, A=A, B=B, q=[0, 0], gaussian=dict(variance=[0.01, 0.01], cells=[31, 31]),
                                state_variables=['x1', 'x2'])
        self.plot_dimensions = [0, 1]

        self.input_space, self.input_grid = Box([-1.0], [1.0]), [5]
        self.state_space = Box([-10, -10], [10, 10], names=self.state_variables)
        self.state_grid = [40, 40]
        self.goal = [Box([-8, -8], [8, 8])]
        self.critical = []
        self.spec, self.horizon = 'reach_avoid', 5
        self.x0 = np.array([9.25, -5.75])

        self.rl_config = RLConfig(max_steps=self.horizon, inflation_rate=[(-2, 2), (-2, 2)])


class IntegratorChain3D(AffineDynamics):
    '''
    Three integrators, as defined in ArchCompStochasticModels.jl (integrator_chain(3)), with the specification of
    IMPaCT's 2D variant: reach the target [-8, 8]^3 within 5 steps, without leaving [-10, 10]^3.

    Input u in [-1, 1] with five inputs, as for the 2D variant. Cells of width (0.25, 0.25, 0.5): with IMPaCT's width
    of 0.5 in every dimension, the bound at x0 is 0.16 (SVMDP) and 0.002 (IMDP) instead of 0.89 and 0.30, because x1
    moves only about one cell per step towards the target. ArchCompStochasticModels.jl
    states this specification as exact-time reach-avoid (the target is checked at step 5 only), which is not
    supported; this variant reaches the target at any time within 5 steps.
    '''

    def __init__(self, args):
        A, B = integrator_chain_matrices(3)
        AffineDynamics.__init__(self, args, A=A, B=B, q=[0, 0, 0],
                                gaussian=dict(variance=[0.01, 0.01, 0.01], cells=[9, 9, 9]),
                                state_variables=['x1', 'x2', 'x3'])
        self.plot_dimensions = [0, 1]

        self.input_space, self.input_grid = Box([-1.0], [1.0]), [5]
        self.state_space = Box([-10, -10, -10], [10, 10, 10], names=self.state_variables)
        self.state_grid = [80, 80, 40]
        self.goal = [Box([-8, -8, -8], [8, 8, 8])]
        self.critical = []
        self.spec, self.horizon = 'reach_avoid', 5
        self.x0 = np.array([9.25, -5.75, -0.25])

        self.rl_config = RLConfig(max_steps=self.horizon, inflation_rate=[(-2, 2), (-2, 2), (-2, 2)])


class IntegratorChain3D_safety(AffineDynamics):
    '''
    PRoTECT's ARCH-COMP25 verification variant: three integrators without control, stay within [-10.1, 10.1]^3 for
    5 steps. As in PRoTECT's equations, the input term is applied with u = 1, i.e., as the constant drift
    (Ns^3/6, Ns^2/2, Ns) (the report's text says u = 0, its equations u = 1).

    PRoTECT certifies the initial set [-10, 10]^3; this pipeline reports the bound at a single x0, the center of
    that set. The grid has 41 cells per dimension, so x0 is the center of a cell.
    Sources: ARCH-COMP25 report (arXiv:2506.17602), Sec. "Benchmarking PRoTECT"; PRoTECT's IC_verification.py.
    '''

    def __init__(self, args):
        A, B_u = integrator_chain_matrices(3)
        B = no_control(self, 3)
        AffineDynamics.__init__(self, args, A=A, B=B, q=B_u[:, 0],
                                gaussian=dict(variance=[0.01, 0.01, 0.01], cells=[9, 9, 9]),
                                state_variables=['x1', 'x2', 'x3'])
        self.plot_dimensions = [0, 1]

        self.state_space = Box([-10.1, -10.1, -10.1], [10.1, 10.1, 10.1], names=self.state_variables)
        self.state_grid = [41, 41, 41]
        self.goal = []
        self.critical = []
        self.spec, self.horizon = 'safety', 5
        self.x0 = np.array([0.0, 0.0, 0.0])
