'''
This is the main Python file for DynAbs-JAX.
The file can be run from the terminal as

```Python3 RunFile.py --model <model-name> ...```

For all available arguments, please see the function :func:`core.options.parse_arguments`.
'''
# %%
# import sys
# sys.argv = [
#     "RunFile.py",
#     "--model", "Dubins_small",
#     "--rl"
# ]

# %%
import datetime
import os
import time
from pathlib import Path
import jax
import numpy as np
import jax.numpy as jnp
import sys

import benchmarks
from core.Gaussian_probabilities import compute_probability_intervals
from core.actions_forward import RectangularForward
from core.model import parse_linear_model, parse_nonlinear_model
from core.options import parse_arguments
from core.partition import RectangularPartition
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import SAC
from RL.RL_Environment import Env
from RL.generate_action_sets import L_infinity, Spheres
import RL.Reward_Evaluate
from RL.run_agents import Agents
from core.imdp import IMDP
import RL.Evaluate_Secondary

# Uncomment one of the following lines to run an example benchmark.
# If it seems to be 'stuck' when computing the transition probabilities, consider decreasing the batch size (e.g., to 1000).
# sys.argv = ['RunFile.py', '--model', 'Dubins_small', '--batch_size', '30000']
# sys.argv = ['RunFile.py', '--model', 'Pendulum', '--batch_size', '30000']
# sys.argv = ['RunFile.py', '--model', 'MountainCar', '--batch_size', '1000', '--plot_title']
# sys.argv = ['RunFile.py', '--model', 'DoubleIntegrator', '--batch_size', '30000', '--plot_title']
# sys.argv = ['RunFile.py', '--model', 'Drone3D_small', '--batch_size', '100', '--plot_title']
# sys.argv = ['RunFile.py', '--model', 'Drone3D', '--batch_size', '10000', '--plot_title']
# sys.argv = ['RunFile.py', '--model', 'Drone2D', '--batch_size', '1000', '--plot_title']

if __name__ == '__main__':
    jax.config.update("jax_default_matmul_precision", "high")

    args = parse_arguments()
    args.floatprecision = np.float32
    if args.gpu:
        jax.config.update('jax_platform_name', 'gpu')
        print('- Requested to run on GPU')
    else:
        jax.config.update('jax_platform_name', 'cpu')
        print('- Requested to run on CPU')
    if args.gpu_rvi:
        args.rvi_device = jax.devices('gpu')[0]
        print('- Requested to run RVI on GPU')
    else:
        args.rvi_device = jax.devices('cpu')[0]
        print('- Requested to run RVI on CPU')

    print('=== JAX STATUS ===')
    print(f'Devices available: {jax.devices()}')
    from jax.extend.backend import get_backend

    print(f'Jax runs on: {get_backend().platform}')
    print('==================\n')

    np.random.seed(args.seed)
    args.jax_key = jax.random.PRNGKey(args.seed)

    # In debug mode, configure jax to use Float64 (for more accurate computations)
    if args.debug:
        from jax import config

        config.update("jax_enable_x64", True)

    # Set current working directory
    args.cwd = os.path.dirname(os.path.abspath(__file__))
    args.root_dir = Path(args.cwd)

    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f'Run started at {stamp} using arguments:')
    for key, val in vars(args).items():
        print(' - `' + str(key) + '`: ' + str(val))
    print('\n==============================\n')

    # Define and parse model
    if args.model == 'Dubins':
        base_model = benchmarks.Dubins(args)
    elif args.model == 'Dubins_small':
        base_model = benchmarks.Dubins_small(args)
    elif args.model == 'Drone2D':
        base_model = benchmarks.Drone2D(args)
    elif args.model == 'Drone3D':
        base_model = benchmarks.Drone3D(args)
    elif args.model == 'Drone3D_small':
        base_model = benchmarks.Drone3D_small(args)
    elif args.model == 'Pendulum':
        base_model = benchmarks.Pendulum(args)
    elif args.model == 'MountainCar':
        base_model = benchmarks.MountainCar(args)
    elif args.model == 'DoubleIntegrator':
        base_model = benchmarks.DoubleIntegrator(args)
    else:
        assert False, f"The passed model '{args.model}' could not be found"

    # decide if we are to use the RL setup, where we have a single abstract action map to a set of concrete actions 
    if args.rl:
        reinforcement_learning = True
    else:
        reinforcement_learning = False

    t = time.time()

    # Parse given model
    if base_model.linear:
        model = parse_linear_model(base_model)
    else:
        model = parse_nonlinear_model(base_model)

    # Create partition of the continuous state space into convex polytope
    partition = RectangularPartition(model=model)
    
    # Create actions based on forward reachable sets
    if reinforcement_learning:
        # TODO - think about how better to construct the radii - these should take into account the magnitude of each component on the action space that we expect so that the spheres are of the approrpriate size
        # NOTE - if the radii are too large then we get really poor satisfaction probability 
        action_dim = model.p

        # TODO - incorporate the new spheres stuff from below up here!!!

                # TODO - bit hard coded for the Dubins_small example
        # define the spheres here, including what dimensions need wrapping and what ones need clipping
        thresholds = jnp.array([4,3,2,1,0])
        radii_options = jnp.array([
            [jnp.pi*0.2, 0.3],
            [jnp.pi*0.1, 0.2],
            [jnp.pi*0.1, 0.1],
            [jnp.pi*0.05, 0.05],
            [0,0]
        ])

        vals_to_clip = [[-np.pi*0.5,np.pi*0.5],[-3,3]]
        vals_to_wrap = [None,None]
        spheres = Spheres(
            thresholds=thresholds,
            radii_options=radii_options,
            vals_to_clip=vals_to_clip,
            vals_to_wrap=vals_to_wrap,
            critical_regions=model.critical
        )

        # sphere_radius = 0.1
        # radii = np.full(action_dim, sphere_radius) # if large can also make the process really slow 
        # with open("results.txt", "a") as f:
        #     f.write(f"Sphere Radius: {sphere_radius}\n")
        actions = RectangularForward(args=args, partition=partition, model=model, action_spheres=spheres)     
        actions_inputs = actions.id_to_input   
    else:
        actions = RectangularForward(args=args, partition=partition, model=model)
        actions_inputs = actions.id_to_input


    # TODO - edit this function to use the new probability intervals 

    P_full, S_id, A_id, P_absorbing = compute_probability_intervals(args=args, 
                                                                    model=model, 
                                                                    partition=partition, 
                                                                    actions=actions,
                                                                    vectorized=True)
    
    del actions

    imdp = IMDP(partition=partition,
                states=np.array(partition.regions['idxs']),
                actions_inputs=actions_inputs,
                x0=model.x0,
                goal_regions=np.array(partition.goal['bools']),
                critical_regions=np.array(partition.critical['bools']),
                P_full=P_full,
                S_id=S_id,
                A_id=A_id,
                P_absorbing=P_absorbing)

    print(f'- Generating abstraction took: {(time.time() - t):.3f} sec.')

    # %% Build and verify with JAX-based RVI

    from core.imdp import RVI_JAX, RVI

    print('Compute optimal policy via robust value iteration with JAX...')

    with jax.default_device(args.rvi_device):

        t = time.time()
        V, _, policy, policy_inputs = RVI_JAX(
            args=args, 
            imdp=imdp, 
            s0=partition.x2state(model.x0)[0], 
            max_iterations=100, 
            epsilon=1e-6, 
            RND_SWEEPS=True, 
            BATCH_SIZE=1000, 
            policy_iteration=True)
        print (f'- RVI with JAX (random-batched asynchronous) took: {(time.time() - t):.3f} sec.')

    # %% Training of the Reinforcement Learning Agent
    if reinforcement_learning:
        '''
        define the reward/evaluation pairs we want to use
        '''

        reward_evals = dict()
        # reward_evals['minimise_action_costs'] = RL.Reward_Evaluate.ActionCosts(np.array([0,-1])) # use 0 to not tax the angle in the input
        # reward_evals['maximise_action_costs'] = RL.Reward_Evaluate.ActionCosts(np.array([0,1])) # use 0 to not tax the angle in the input
        # reward_evals['get_close_bottom_right'] = RL.Reward_Evaluate.GetCloseToArea(region_lower=np.array([10,-10]), region_upper=np.array([10,-10]), dims=[0,1])
        # reward_evals['get_close_vertical_critical'] = RL.Reward_Evaluate.GetCloseToArea(region_lower=np.array([-1,-5]), region_upper=np.array([1,4]), dims=[0,1])
        # reward_evals['get_closer_than_base_to_bottom_right'] = RL.Reward_Evaluate.GetCloserThanBaseToArea(region_lower=np.array([10,-10]), region_upper=np.array([10,-10]), dims=[0,1])
        reward_evals['get_closer_than_base_to_top_right'] = RL.Reward_Evaluate.GetCloserThanBaseToArea(region_lower=np.array([10,10]), region_upper=np.array([10,10]), dims=[0,1])
        # reward_evals['get_closer_than_base_to_bottom_left'] = RL.Reward_Evaluate.GetCloserThanBaseToArea(region_lower=np.array([-10,-10]), region_upper=np.array([-10,-10]), dims=[0,1])
        # reward_evals['get_closer_than_base_to_vertical_critical'] = RL.Reward_Evaluate.GetCloserThanBaseToArea(region_lower=np.array([-1,-5]), region_upper=np.array([1,4]), dims=[0,1])
        reward_evals['get_closer_than_base_to_top_opening'] = RL.Reward_Evaluate.GetCloserThanBaseToArea(region_lower=np.array([-1,6.5]), region_upper=np.array([-1,6.5]), dims=[0,1])
        reward_evals['top_opening_double_reward'] = RL.Reward_Evaluate.GetToRegionDoubleReward(region1_lower=np.array([-1,6.5]), region1_upper=np.array([-1,6.5]), region2_lower=np.array([10,10]), region2_upper=np.array([10,10]), dims=[0,1])
        reward_evals['top_opening_double_reward'] = RL.Reward_Evaluate.GetToRegionDoubleReward(region2_lower=np.array([-1,6.5]), region2_upper=np.array([-1,6.5]), region1_lower=np.array([10,10]), region1_upper=np.array([10,10]), dims=[0,1])


        print("Constructing RL Environment")
        # vectorize the environment
        init_env = lambda reward_structure: make_vec_env(
                lambda: Env(
                    state_dim=model.n,
                    space_lower=model.partition['boundary'][0],
                    space_upper=model.partition['boundary'][1],
                    action_dim=model.p,
                    action_lower=model.uMin,
                    action_upper=model.uMax,
                    initial_state=model.x0,
                    model=model,
                    policy_inputs=policy_inputs,
                    spheres=spheres,
                    reward_structure=reward_structure,
                    partition=partition
                ),
                n_envs=1
            )
        
        agents = Agents(reward_evals=reward_evals, init_env=init_env, timesteps = 250_000)

        # train all the agents
        agents.train_agents(dont_train=args.no_train)


    # %% Simulations

    sim_policy = policy
    sim_policy_inputs = policy_inputs
    sim_values = V

    from core.simulate import MonteCarloSim
    from RL.helper_functions import run_simulations, plot


    # NOTE: currently set up for the Mountain Car
    # scaling = np.array([100])
    # evaluation = EnergyEfficiency(action_scaling=scaling)
    # evaluation = DistanceToRegion(region_lower=np.array([-7,1], dtype=float), region_upper=np.array([-1,3], dtype=float), dims=[0,2])
    # evaluation = TimeSteps()

    # TODO - currently we only support infinite horizons - we should extend this to finite horizons

    # define if we want to check if the model enteres a given box (None if we don't)
    tracked_region = np.array([[-0.5, 2, -np.pi],[3,10,np.pi]])
    if reinforcement_learning:
        agent_envs = agents.get_agents_envs_evals()

        run_simulations(
            agent_envs=agent_envs,
            model=model,
            args=args,
            stamp=stamp,
            partition=partition,
            policy=policy,
            policy_inputs=policy_inputs,
            sim_values=sim_values,
            spheres=spheres,
            show_plot=False,
            tracked_region=tracked_region     # define the region to track entering (if we want to)
        )

    else:
        sim = MonteCarloSim(model, partition, sim_policy, sim_policy_inputs, model.x0, verbose=False, iterations=1000,tracked_region=tracked_region)
        print('Empirical satisfaction probability:', sim.results['satprob'])
        if tracked_region is not None:
            print(f'Times entered tracked region: {sim.results['tracked_region']}')
        plot(sim,model,args,stamp,partition,sim_values,sim_policy_inputs)

    # %% Plot

    # plot(sim)
    # if reinforcement_learning:
    #     print('PLOTTING RL')
    #     # TODO - plotting for the RL trace doesn't work...
    #     plot(sim_rl, rl=True)
    # print('DONE')
        
# %%
