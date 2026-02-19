import numpy as np
from core.simulate import MonteCarloSim
from plotting.traces import plot_traces
from plotting.heatmap import heatmap
import RL.Evaluate_Secondary

# given an action proposed in the hyperrectangle [(-1,...,-1), (1,...,1)], find the corresponding real concrete action by scaling appropriately
def project_action(action, action_lower, action_upper):
	result = action_lower + (action + 1) * (action_upper - action_lower) / 2
	return result

# plot the traces of a simulation
def plot(sim, model, args, stamp, partition, sim_values, sim_policy_inputs, filename="traces", show_plot = True):
	plot_traces(args, stamp, model.plot_dimensions, partition, model, sim.results['traces'], line=False, num_traces=10, add_unsafe_box=False,filename=filename,show_plot=show_plot)
	heatmap(args, stamp, idx_show=model.plot_dimensions, slice_values=np.zeros(model.n), partition=partition, results=sim_values, filename="heatmap_satprob", show_plot=show_plot)
	heatmap(args, stamp, idx_show=model.plot_dimensions, slice_values=np.zeros(model.n), partition=partition, results=sim_policy_inputs[:,0], filename="heatmap_inputs", show_plot=show_plot)
	
	if args.model == 'Pendulum':
		model.plot_trajectory_gif(np.array(sim.results['traces'][0]['x'])[:,0], filename=f'output/pendulum_{stamp}.gif')

	if args.model == 'MountainCar':
		model.plot_trajectory_gif(np.array(sim.results['traces'][0]['x'])[:,0], filename=f'output/mountaincar_{stamp}.gif')


def run_simulations(agent_envs, model, args, stamp, partition, policy, policy_inputs, sim_values, derive_set, iterations=1000, verbose=False, show_plot=True):
	'''
	Docstring for run_simulations
	
	:param agent_envs: list of tuples containing (name, agent, environment, evaluation structure)
	:param model: dynamic model in question
	:param partition: partition of the model's action and state space
	:param policy: optimal policy found via value/policy iteration 
	:param policy_inputs: policy inputs
	:param verbose: as it says on the tin
	:param iterations: how many times should we run each simulation
	'''

	# TODO - maybe we want to run the sims without the RL agent on the policy that was synthesised without spheres as this is the status quo? 
	with open("results.txt", "a") as f:
		for (s,agent,vecnorm,evaluation) in agent_envs:
			f.write(f'Doing simulation for {s}\n')
			# run sims without RL agent
			sim = MonteCarloSim(model, partition, policy, policy_inputs, model.x0, project_action, verbose=False, iterations=100, evaluate_secondary=evaluation)
			f.write(f'Average Secondary Score: {sim.results["secondary_score"]}\n')
			f.write(f'Empirical satisfaction probability: {sim.results['satprob']}\n')

			if isinstance(evaluation, RL.Evaluate_Secondary.DistanceToRegion):
				f.write(f'closest = {evaluation.closest}\n')
				evaluation.closest = float('inf')

			# run sims with RL agent
			sim_rl = MonteCarloSim(model, partition, policy, policy_inputs, model.x0, project_action, verbose=False, iterations=100, evaluate_secondary=evaluation, agent=agent, derive_set=derive_set, vecnorm=vecnorm)
			f.write(f"Average Secondary Score with RL agent: {sim_rl.results['secondary_score']}\n")
			f.write(f"Empirical satisfaciton probability with RL agent: {sim_rl.results['satprob']}\n")
			if isinstance(evaluation, RL.Evaluate_Secondary.DistanceToRegion):
				f.write(f'closest = {evaluation.closest}\n')
				evaluation.closest = float('inf')
			
			f.write('\n')

			plot(
				sim=sim,
				model=model,
				args=args,
				stamp=stamp,
				partition=partition,
				sim_values=sim_values,
				sim_policy_inputs=policy_inputs,
				filename=s+'_NO_RL',
				show_plot=show_plot
			)

			plot(
				sim=sim_rl,
				model=model,
				args=args,
				stamp=stamp,
				partition=partition,
				sim_values=sim_values,
				sim_policy_inputs=policy_inputs,
				filename=s+'_RL',
				show_plot=show_plot
			)