'''
This is the main Python file for DynAbs-JAX.
The file can be run from the terminal as

```Python3 RunFile.py --model <model-name> ...```

For all available arguments, please see the function :func:`core.options.parse_arguments`.
'''
# %%
import sys
sys.argv = [
    "RunFile.py",
    "--model", "Dubins_small",
    "--rl"
]

# %%
import datetime
import os
import time
from pathlib import Path

import jax
import numpy as np

import benchmarks
from core.Gaussian_probabilities import compute_probability_intervals
from core.actions_forward import RectangularForward
from core.model import parse_linear_model, parse_nonlinear_model
from core.options import parse_arguments
from core.partition import RectangularPartition
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import SAC
from RL.RL_Environment import Env
from RL.generate_action_sets import L_infinity
import RL.Reward_Evaluate
from RL.run_agents import Agents

# import sys
# sys.argv = ['RunFile.py', '--model', 'Dubins_small', '--batch_size', '30000']
# sys.argv = ['RunFile.py', '--model', 'Pendulum', '--batch_size', '30000']
# sys.argv = ['RunFile.py', '--model', 'MountainCar', '--batch_size', '30000']

if __name__ == '__main__':
    jax.config.update("jax_default_matmul_precision", "high")

    args = parse_arguments()
    if args.gpu:
        jax.config.update('jax_platform_name', 'gpu')
    else:
        jax.config.update('jax_platform_name', 'cpu')

    print('=== JAX STATUS ===')
    # print(f'Devices available: {jax.devices()}')
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
    elif args.model == 'Pendulum':
        base_model = benchmarks.Pendulum(args)
    elif args.model == 'MountainCar':
        base_model = benchmarks.MountainCar(args)
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
    print(f"(Number of states: {len(partition.regions['idxs'])})\n")

    # Create actions based on forward reachable sets
    if reinforcement_learning:
        # TODO - think about how better to construct the radii - these should take into account the magnitude of each component on the action space that we expect so that the spheres are of the approrpriate size
        # NOTE - if the radii are too large then we get really poor satisfaction probability 
        action_dim = model.p
        radii = np.full(action_dim, 0) # if large can also make the process really slow 
        actions = RectangularForward(partition=partition, model=model, action_sets=reinforcement_learning, radii=radii)        
    else:
        actions = RectangularForward(partition=partition, model=model)

    # With forward reachability, every action is enabled in every state
    enabled_actions = np.full((len(partition.regions['centers']), len(actions.idxs)), fill_value=True, dtype=np.bool)

    print(f"(Number of actions in each state: {np.sum(np.any(enabled_actions, axis=0))})\n")

    # TODO - edit this function to use the new probability intervals 
    P_full, P_id, P_absorbing = compute_probability_intervals(args, model, partition, actions.frs, actions.max_slice)

    # %% Model checking

    from core.imdp import BuilderStorm

    # Compute optimal policy on the iMDP abstraction
    print('\nCreate iMDP using storm...')

    # Build interval MDP via StormPy
    builderS = BuilderStorm(partition=partition,
                            actions=actions,
                            states=np.array(partition.regions['idxs']),
                            x0=model.x0,
                            goal_regions=np.array(partition.goal['idxs']),
                            critical_regions=np.array(partition.critical['idxs']),
                            P_full=P_full,
                            P_id=P_id,
                            P_absorbing=P_absorbing)

    print(f'- Generating abstraction took: {(time.time() - t):.3f} sec.')
    print(builderS.imdp)

    t = time.time()
    result = builderS.compute_reach_avoid()
    policy, policy_inputs = builderS.get_policy(actions)
    print(f'- Verify with storm took: {(time.time() - t):.3f} sec.')
    print('Total sum of reach probs:', np.sum(builderS.results))
    print('Value in state {}: {}'.format(model.x0, builderS.get_value_from_tuple(model.x0, partition)))

    # %% Training of the Reinforcement Learning Agent
    if reinforcement_learning:
        '''
        define the reward/evaluation pairs we want to use
        '''

        reward_evals = dict()
        reward_evals['minimise_action_costs'] = RL.Reward_Evaluate.ActionCosts(np.array([0,-1])) # use 0 to not tax the angle in the input
        reward_evals['maximise_action_costs'] = RL.Reward_Evaluate.ActionCosts(np.array([0,-1])) # use 0 to not tax the angle in the input
        reward_evals['get_close_bottom_right'] = RL.Reward_Evaluate.GetCloseToArea(region_lower=np.array([-10,-10]), region_upper=np.array([-10,-10]), dims=[0,1])

        # add an extra critical region that the formal verification is unaware of
        # if args.model == 'Dubins_small':
        #     model.critical = np.append(model.critical, [[[5,0.5,-np.pi],[8,1,np.pi]]], axis=0)
        #     partition = RectangularPartition(model=model)

        derive_set = lambda centre: L_infinity(centre,radii,model.uMin,model.uMax)

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
                    derive_set=derive_set,
                    reward_structure=reward_structure,
                    partition=partition
                ),
                n_envs=1
            )
        
        agents = Agents(reward_evals=reward_evals, init_env=init_env, timesteps = 10_000)

        # train all the agents
        agents.train_agents()


    # %% Simulations

    from core.simulate import MonteCarloSim
    from plotting.traces import plot_traces
    from plotting.heatmap import heatmap

    # NOTE: currently set up for the Mountain Car
    # scaling = np.array([100])
    # evaluation = EnergyEfficiency(action_scaling=scaling)
    # evaluation = DistanceToRegion(region_lower=np.array([-7,1], dtype=float), region_upper=np.array([-1,3], dtype=float), dims=[0,2])
    # evaluation = TimeSteps()

    # TODO - currently we only support infinite horizons - we should extend this to finite horizons
    if reinforcement_learning:
        agent_envs = agents.get_agents_envs_evals()

        # TODO - maybe we want to run the sims without the RL agent on the policy that was synthesised without spheres as this is the status quo? 
        for (agent,vecnorm,evaluation) in agent_envs:
            # run sims without RL agent
            sim = MonteCarloSim(model, partition, policy, policy_inputs, model.x0, verbose=False, iterations=100, evaluate_secondary=evaluation)
            print(f'Average Secondary Score: {sim.results["secondary_score"]}')
            print('Empirical satisfaction probability:', sim.results['satprob'])

            # run sims with RL agent
            sim_rl = MonteCarloSim(model, partition, policy, policy_inputs, model.x0, verbose=False, iterations=100, evaluate_secondary=evaluation, agent=agent, derive_set=derive_set, vecnorm=vecnorm)
            print(f"Average Secondary Score with RL agent: {sim_rl.results['secondary_score']}")
            print(f"Empirical satisfaciton probability with RL agent: {sim_rl.results['satprob']}")

    else:
        sim = MonteCarloSim(model, partition, policy, policy_inputs, model.x0, verbose=False, iterations=100)
        print('Empirical satisfaction probability:', sim.results['satprob'])

    # %% Plot
    def plot(sim, rl=False):
        plot_traces(args, stamp, model.plot_dimensions, partition, model, sim.results['traces'], line=False, num_traces=10, add_unsafe_box=False,)
        if rl:
            heatmap(args, stamp, idx_show=model.plot_dimensions, slice_values=np.zeros(model.n), partition=partition, results=builderS.results, filename="rl_heatmap_satprob")
        else:
            heatmap(args, stamp, idx_show=model.plot_dimensions, slice_values=np.zeros(model.n), partition=partition, results=builderS.results, filename="heatmap_satprob")
        if model.p >  1:
            if rl:
                heatmap(args, stamp, idx_show=model.plot_dimensions, slice_values=np.zeros(model.n), partition=partition, results=policy_inputs[:,0], filename="rl_heatmap_inputs")
            else:
                heatmap(args, stamp, idx_show=model.plot_dimensions, slice_values=np.zeros(model.n), partition=partition, results=policy_inputs[:,0], filename="heatmap_inputs")
        # TODO - uncomment this - seems broken for now for Mountain Car at least?
        # else:
        #     if rl:
        #         heatmap(args, stamp, idx_show=model.plot_dimensions, slice_values=np.zeros(model.n), partition=partition, results=policy_inputs, filename="rl_heatmap_inputs")            
        #     else:
        #         heatmap(args, stamp, idx_show=model.plot_dimensions, slice_values=np.zeros(model.n), partition=partition, results=policy_inputs, filename="heatmap_inputs")

        if args.model == 'Pendulum':
            model.plot_trajectory_gif(np.array(sim.results['traces'][0]['x'])[:,0], filename=f'output/pendulum_{stamp}.gif')

        if args.model == 'MountainCar':
            model.plot_trajectory_gif(np.array(sim.results['traces'][0]['x'])[:,0], filename=f'output/mountaincar_{stamp}.gif')

    # plot(sim)
    # if reinforcement_learning:
    #     print('PLOTTING RL')
    #     # TODO - plotting for the RL trace doesn't work...
    #     plot(sim_rl, rl=True)
    # print('DONE')
