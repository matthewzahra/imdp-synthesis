'''
Automated anaesthesia (AS) benchmark of the ARCH-COMP stochastic models category (ARCH-COMP18, Sec. 2.1), in its
fully automated form (no anaesthesiologist), as a three-compartment pharmacokinetic model:

    x[k+1] = A x[k] + B v[k] + w[k],   w ~ N(0, 0.001 I),   v in [0, 7].

The noise variance 0.001 is that of ArchCompStochasticModels.jl (fully_automated_anaesthesia) and of IMPaCT's and
PRoTECT's ARCH-COMP25 code. The safe set is [1, 6] x [0, 10]^2 in all variants; it is the partition boundary, so
leaving it counts as a failure.

Grids: x2 and x3 change by at most about 0.1 per step, so their cells must be narrow. Within a cell, the abstraction
lets the worst case move to any cell that the successors overlap, i.e., by up to one cell per step. With the cells of
0.5 to 1 wide that IntervalMDP.jl and IMPaCT used, this spurious drift reaches the boundary within a few steps, and the
bound at x0 is about 0 (for example, 7e-6 instead of 0.88 for Anaesthesia_safety on the SVMDP path). All variants
therefore use cells of width 0.1 in x2 and x3.

The full benchmark with the anaesthesiologist's bolus decisions is a hybrid system (512 discrete modes) and is not
supported.
'''

import numpy as np

from benchmarks.archcomp.common import boxes, no_control, set_grid
from benchmarks.models import AffineDynamics
from core.rl.config import RLConfig

AS_A = [[0.8192, 0.03412, 0.01265],
        [0.01646, 0.9822, 0.0001],
        [0.0009, 0.00002, 0.9989]]
AS_B = [[0.01883], [0.0002], [0.00001]]
AS_NOISE_VAR = [0.001, 0.001, 0.001]
AS_SAFE_SET = [[1, 0, 0], [6, 10, 10]]
AS_STATE_VARIABLES = ['c_central', 'c_peripheral', 'c_effect']


class Anaesthesia_safety(AffineDynamics):
    '''
    IntervalMDP.jl's ARCH-COMP25 variant: stay in the safe set for 10 steps, with the inputs {0, 3.5, 7} of that entry
    (which reports a maximal lower bound of 99.98% and a mean of 34.31%, on a grid of (12, 20, 20) cells). Grid of
    (12, 100, 100) cells (see the module docstring). The benchmark defines no initial state; x0 is the center of
    PRoTECT's initial set [4, 6] x [8, 10]^2.
    Sources: ARCH-COMP25 report (arXiv:2506.17602), Sec. "Benchmarking IntervalMDP.jl"; ArchCompStochasticModels.jl.
    '''

    def __init__(self, args):
        AffineDynamics.__init__(self, args, A=AS_A, B=AS_B, q=[0, 0, 0], noise_var=AS_NOISE_VAR,
                                noise_cells=[5, 5, 5], state_variables=AS_STATE_VARIABLES)
        self.plot_dimensions = [1, 2]

        self.uMin, self.uMax, self.num_actions = [0.0], [7.0], [3]
        set_grid(self, AS_SAFE_SET, [12, 100, 100])
        self.goal = boxes()
        self.critical = boxes()
        self.spec, self.horizon = 'safety', 10
        self.x0 = np.array([5.0, 9.0, 9.0])


class Anaesthesia_reach(AffineDynamics):
    '''
    IMPaCT's ARCH-COMP25 variant: reach [4, 6] x [8, 10]^2 within 10 steps, without leaving the safe set. As in
    IMPaCT's code, the anaesthesiologist's bolus is a second input: u = (v, bolus) in [0, 7] x {0, 30}, which enter
    as B (v + bolus). Inputs spaced (1, 30) apart, as in IMPaCT's entry; cells of width (0.25, 0.1, 0.1) instead of
    IMPaCT's (0.25, 1, 1) (see the module docstring). The benchmark defines no initial state; from x0 the target is
    reachable within 10 steps. The IMDP path needs about 5.5 GB of memory.
    Sources: ARCH-COMP25 report (arXiv:2506.17602), Sec. "Benchmarking IMPaCT"; IMPaCT's AS.cpp.
    '''

    def __init__(self, args):
        B = np.hstack([AS_B, AS_B])
        AffineDynamics.__init__(self, args, A=AS_A, B=B, q=[0, 0, 0], noise_var=AS_NOISE_VAR,
                                noise_cells=[5, 5, 5], state_variables=AS_STATE_VARIABLES)
        self.plot_dimensions = [0, 1]

        self.uMin, self.uMax, self.num_actions = [0.0, 0.0], [7.0, 30.0], [8, 2]
        set_grid(self, AS_SAFE_SET, [20, 100, 100])
        self.goal = boxes([[4, 8, 8], [6, 10, 10]])
        self.critical = boxes()
        self.spec, self.horizon = 'reach_avoid', 10
        self.x0 = np.array([2.125, 9.5, 9.5])

        self.rl_config = RLConfig(max_steps=self.horizon, inflation_rate=[(-2, 2), (-1, 1), (-1, 1)])


class Anaesthesia_verification(AffineDynamics):
    '''
    PRoTECT's ARCH-COMP25 verification variant: no input (v = 0), stay in the safe set for 10 steps. PRoTECT
    certifies the initial set [4, 6] x [8, 10]^2; this pipeline reports the bound at a single x0, its center. Grid of
    (20, 100, 100) cells (see the module docstring).
    Sources: ARCH-COMP25 report (arXiv:2506.17602), Sec. "Benchmarking PRoTECT"; PRoTECT's AS_verification.py.
    '''

    def __init__(self, args):
        B = no_control(self, 3)
        AffineDynamics.__init__(self, args, A=AS_A, B=B, q=[0, 0, 0], noise_var=AS_NOISE_VAR,
                                noise_cells=[5, 5, 5], state_variables=AS_STATE_VARIABLES)
        self.plot_dimensions = [1, 2]

        set_grid(self, AS_SAFE_SET, [20, 100, 100])
        self.goal = boxes()
        self.critical = boxes()
        self.spec, self.horizon = 'safety', 10
        self.x0 = np.array([5.0, 9.0, 9.0])
