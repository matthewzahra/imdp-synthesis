'''
Building automation system (BA), case study CS1 (4D), of the ARCH-COMP stochastic models category (ARCH-COMP19,
Sec. 4.2):

    x[k+1] = A x[k] + B u[k] + Q + w[k],   w ~ N(0, diag(0.0774, 0.0774, 0.3872, 0.3098)),

where x1 and x2 are the temperatures of two zones and x3 and x4 those of their radiators. The matrices are those of
ArchCompStochasticModels.jl (cs1_bas). IMPaCT's and PRoTECT's ARCH-COMP25 code use Q1 = 3.4378 instead of 3.3378.

All variants are safety specifications over 6 steps. The partition boundary is the safe set; for dimensions that
the specification leaves unconstrained, it is a region of interest, and leaving it counts as a failure (which is
conservative). The benchmark defines no initial state; x0 is the setpoint of both zones with the radiators near
their steady state.

PRoTECT's verification variant (u = 0) is not included: its initial set lies outside its state space, both in the
report and in PRoTECT's code.
'''

import numpy as np

from benchmarks.affine import AffineDynamics
from core.box import Box

CS1_A = [[0.6682, 0.0,    0.02632,   0.0],
         [0.0,    0.6830, 0.0,       0.02096],
         [1.0005, 0.0,    -0.000499, 0.0],
         [0.0,    0.8004, 0.0,       0.1996]]
CS1_B = [[0.1320], [0.1402], [0.0], [0.0]]
CS1_Q = [3.3378, 2.9272, 13.0207, 10.4166]
CS1_NOISE_VAR = [0.0774, 0.0774, 0.3872, 0.3098]


class _BuildingAutomation4D(AffineDynamics):
    '''CS1 dynamics with a safety specification over 6 steps on the partition boundary.'''

    def __init__(self, args, boundary, number_per_dim, u_bounds, num_actions, noise_cells):
        AffineDynamics.__init__(self, args, A=CS1_A, B=CS1_B, q=CS1_Q,
                                gaussian=dict(variance=CS1_NOISE_VAR, cells=noise_cells),
                                state_variables=['T_zone1', 'T_zone2', 'T_rad1', 'T_rad2'])
        self.plot_dimensions = [0, 1]

        self.input_space, self.input_grid = Box([u_bounds[0]], [u_bounds[1]]), [num_actions]
        self.state_space = Box(*boundary, names=self.state_variables)
        self.state_grid = number_per_dim
        self.goal = []
        self.critical = []
        self.spec, self.horizon = 'safety', 6
        self.x0 = np.array([20.0, 20.0, 33.0, 33.0])
        # No rl_config: safety specifications run with --dense


class BuildingAutomation4D(_BuildingAutomation4D):
    '''
    IntervalMDP.jl's ARCH-COMP25 variant: keep both zone temperatures within 20 +- 0.5 degrees Celsius; radiators in
    the region of interest [30, 36]^2. Grid of (5, 5, 7, 7) cells and four inputs evenly spaced in [15, 22], as in that
    entry, which reports a maximal lower bound of 8.12% and a mean of 6.20% over the states.
    Sources: ARCH-COMP25 report (arXiv:2506.17602), Sec. "Benchmarking IntervalMDP.jl"; ArchCompStochasticModels.jl.
    '''

    def __init__(self, args):
        super().__init__(args, boundary=[[19.5, 19.5, 30, 30], [20.5, 20.5, 36, 36]], number_per_dim=[5, 5, 7, 7],
                         u_bounds=(15.0, 22.0), num_actions=4, noise_cells=[41, 41, 15, 15])


class BuildingAutomation4D_IMPaCT(_BuildingAutomation4D):
    '''
    IMPaCT's ARCH-COMP25 variant: stay within [19, 21]^2 x [30, 36]^2 (the state space of IMPaCT's code; the report's
    text says [19, 20]^2). Inputs 17, 18, 19, 20 and cells of width (0.5, 0.5, 1, 1), as in IMPaCT's grid.
    Sources: ARCH-COMP25 report (arXiv:2506.17602), Sec. "Benchmarking IMPaCT"; IMPaCT's BA_4d.cpp.
    '''

    def __init__(self, args):
        super().__init__(args, boundary=[[19, 19, 30, 30], [21, 21, 36, 36]], number_per_dim=[4, 4, 6, 6],
                         u_bounds=(17.0, 20.0), num_actions=4, noise_cells=[21, 21, 15, 15])


class BuildingAutomation4D_zone1(_BuildingAutomation4D):
    '''
    SySCoRe's ARCH-COMP25 variant: keep only the temperature of zone 1 within 20 +- 0.5 degrees Celsius. Zone 2 and
    the radiators are unconstrained; their regions of interest are [18.5, 21.5] and [30, 36]^2. SySCoRe does not
    state its input space or grid; this variant uses those of BuildingAutomation4D (four inputs in [15, 22]).
    Sources: ARCH-COMP25 report (arXiv:2506.17602), Sec. "Benchmarking SySCoRe".
    '''

    def __init__(self, args):
        super().__init__(args, boundary=[[19.5, 18.5, 30, 30], [20.5, 21.5, 36, 36]], number_per_dim=[5, 6, 7, 7],
                         u_bounds=(15.0, 22.0), num_actions=4, noise_cells=[41, 21, 15, 15])
