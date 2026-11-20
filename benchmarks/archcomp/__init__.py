'''
Benchmarks of the ARCH-COMP friendly competition, category stochastic models, in the variants that the 2025 edition
ran, as far as this pipeline supports them (discrete time, additive Gaussian noise, reach-avoid or safety).

Main source: ARCH-COMP25 Category Report: Stochastic Models (arXiv:2506.17602). Model definitions are cross-checked
with ArchCompStochasticModels.jl (https://github.com/Zinoex/ArchCompStochasticModels.jl) and with the ARCH-COMP25 code
of IMPaCT (examples/ARCH-COMP/2025) and PRoTECT (ex/ARCH-COMP/2025). Each module's docstring lists its sources,
where they disagree, and the choices the benchmark leaves open (such as the initial state).
'''

from .anaesthesia import Anaesthesia_reach, Anaesthesia_safety, Anaesthesia_verification
from .autonomous_vehicle import AutonomousVehicle3D, AutonomousVehicle3D_coarse
from .building_automation import BuildingAutomation4D, BuildingAutomation4D_IMPaCT, BuildingAutomation4D_zone1
from .integrator_chain import IntegratorChain2D, IntegratorChain3D, IntegratorChain3D_safety
from .patrol_robot import PatrolRobot_reach, PatrolRobot_reachavoid
from .van_der_pol import VanDerPol, VanDerPol_SySCoRe

ARCHCOMP_BENCHMARKS = {cls.__name__: cls for cls in (
    Anaesthesia_safety, Anaesthesia_reach, Anaesthesia_verification,
    AutonomousVehicle3D, AutonomousVehicle3D_coarse,
    BuildingAutomation4D, BuildingAutomation4D_IMPaCT, BuildingAutomation4D_zone1,
    IntegratorChain2D, IntegratorChain3D, IntegratorChain3D_safety,
    PatrolRobot_reachavoid, PatrolRobot_reach,
    VanDerPol, VanDerPol_SySCoRe,
)}
