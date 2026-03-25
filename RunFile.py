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
from core.model import parse_linear_model, parse_nonlinear_model
from core.options import parse_arguments
from core.partition import RectangularPartition
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import SAC
from RL.RL_Environment import Env
from RL.run_agents import Agents
from core.imdp import IMDP
from RL.sphere_defs import build_sphere_defs_test, build_sphere_model, generate_clip_wrap_vals
from core.build_policy import build_policy
from RL.Reward_Evaluate import generate_reward_eval
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

    if args.test_spheres:
        sphere_defs = build_sphere_defs_test()

        vals_to_clip,vals_to_wrap = generate_clip_wrap_vals(args.model)

        radii_funcs = [
            lambda d: jnp.where(d < 0.5, 0, ((d**2)/20)*np.pi),
            lambda d: 0
        ]

        for sphere in sphere_defs:
            thresholds, radii_options = sphere.thresholds, sphere.radii

            V, policy, policy_inputs, spheres = build_policy(
                thresholds=thresholds,
                radii_options=radii_options,
                vals_to_clip=vals_to_clip,
                vals_to_wrap=vals_to_wrap,
                model=model,
                partition=partition,
                args=args,
                stamp=stamp,
                t=t,
                reinforcement_learning=reinforcement_learning,
                radii_funcs=radii_funcs,
                continuous=False
            )

            with open(f"{stamp}_results.txt", "a") as f:
                f.write(str(spheres))

        exit()

    else:
        # extract the saved sphere definition for the model that we are running
        sphere_def = build_sphere_model(model_name=args.model)
        thresholds = sphere_def.thresholds
        radii_options = sphere_def.radii

        vals_to_clip,vals_to_wrap = generate_clip_wrap_vals(args.model)

        # generate the policy that takes spheres into account that the RL agent will make use of
        if reinforcement_learning:
            V_rl, policy_rl, policy_inputs_rl, spheres_rl = build_policy(
                thresholds=thresholds,
                radii_options=radii_options,
                vals_to_clip=vals_to_clip,
                vals_to_wrap=vals_to_wrap,
                model=model,
                partition=partition,
                args=args,
                stamp=stamp,
                t=t,
                reinforcement_learning=reinforcement_learning,
                continuous=False
            )
        
        # generate the sphere-less policy
        V, policy, policy_inputs, spheres = build_policy(
            thresholds=thresholds,
            radii_options=radii_options,
            vals_to_clip=vals_to_clip,
            vals_to_wrap=vals_to_wrap,
            model=model,
            partition=partition,
            args=args,
            stamp=stamp,
            t=t,
            reinforcement_learning=False,
            continuous=False
        )

    # %% Training of the Reinforcement Learning Agent
    if reinforcement_learning:
        '''
        define the reward/evaluation pairs we want to use
        '''

        reward_evals = generate_reward_eval()

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
                    spheres=spheres_rl,
                    reward_structure=reward_structure,
                    partition=partition,
                ),
                n_envs=1
            )
        
        agents = Agents(reward_evals=reward_evals, init_env=init_env, timesteps = 250_000)

        # train all the agents
        agents.train_agents(dont_train=args.no_train)


    # %% Simulations

    # sim_policy = policy_rl
    # sim_policy_inputs = policy_inputs_rl
    # sim_values = V_rl

    from core.simulate import MonteCarloSim
    from RL.helper_functions import run_simulations, plot

    agent_envs = None

    # define if we want to check if the model enteres a given box (None if we don't)
    tracked_region = {"Dubins_small": np.array([[1, 2, -np.pi],[3,10,np.pi]])}
    if reinforcement_learning:
        agent_envs = agents.get_agents_envs_evals()

        run_simulations(
            agent_envs=agent_envs,
            model=model,
            args=args,
            stamp=stamp,
            partition=partition,
            policy=policy_rl,
            policy_inputs=policy_inputs_rl,
            sim_values=V_rl,
            spheres=spheres_rl,
            show_plot=False,
            tracked_region=tracked_region.get(args.model, None)     # define the region to track entering (if we want to, None
        )

    # get the evals used (if any)
    if agent_envs is not None:
        with open(f"{stamp}_results.txt", "a") as f:
            f.write("Running Evaluations for the standard NO-RL-NO-SPHERE approach\n")
            for (s,_,_,evaluation) in agent_envs:
                f.write(f'Doing simulation for {s}\n')
                sim = MonteCarloSim(model,partition,policy,policy_inputs,model.x0,verbose=False,iterations=1000,tracked_region=tracked_region.get(args.model,None),evaluate_secondary=evaluation)
                f.write(f'Secondary Score: {sim.results["secondary_score"]}\n')
                f.write(f'Trace Average Secondary Score: {sim.results["secondary_score_average"]}\n')
                f.write(f'Empirical satisfaction probability: {sim.results['satprob']}\n')

                if isinstance(evaluation, RL.Evaluate_Secondary.DistanceToRegion):
                    f.write(f'closest = {evaluation.closest}\n')
                    evaluation.closest = float('inf')

                if tracked_region is not None:
                    f.write(f'Times entered tracked region: {sim.results['tracked_region']}\n')


                f.write(f'Average trace length: {int(np.mean(list(map(lambda trace: len(trace['u']), list(sim.results['traces'].values())))))}\n')
                
                f.write('\n\n')


    else:
        sim = MonteCarloSim(model, partition, policy, policy_inputs, model.x0, verbose=False, iterations=1000,tracked_region=tracked_region.get(args.model,None))
        print('Empirical satisfaction probability:', sim.results['satprob'])
        if tracked_region.get(args.model,None) is not None:
            print(f'Times entered tracked region: {sim.results['tracked_region']}')
        plot(sim,model,args,stamp,partition,V,policy_inputs)

    # %% Plot

    # plot(sim)
    # if reinforcement_learning:
    #     print('PLOTTING RL')
    #     # TODO - plotting for the RL trace doesn't work...
    #     plot(sim_rl, rl=True)
    # print('DONE')
        
# %%
