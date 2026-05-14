import numpy as np
from core.simulate import MonteCarloSim
from plotting.traces import plot_traces
from plotting.heatmap import heatmap
import RL.Evaluate_Secondary
import jax.numpy as jnp
import matplotlib.pyplot as plt

# given an action proposed in the hyperrectangle [(-1,...,-1), (1,...,1)], find the corresponding real concrete action by scaling appropriately
def project_action(action, action_lower, action_upper):
	result = action_lower + (action + 1) * (action_upper - action_lower) / 2
	return result

# plot the magnitude of each action over time for a set of given traces
def plot_action_magnitudes(fname,traces,number=10,max_y=None):
	plt.clf()

	for trace in list(traces.values())[:number]:
		values = trace['u']

		magnitudes = list(map(np.linalg.norm,values))

		timesteps = [i for i in range(len(magnitudes))]

		plt.plot(timesteps, magnitudes)
		
	plt.xlabel("Timesteps")
	plt.ylabel("Control Input Magnitude")
	# plt.axhline(0)

	if max_y:
		plt.ylim(bottom=0, top=max_y)
	else:
		plt.ylim(bottom=0, top=max(1,plt.ylim()[1]))

	# save the figure
	plt.savefig(f"{fname}.png",bbox_inches='tight')

	plt.clf()

# plot the difference in magnitudes between successive actions
def plot_action_magnitude_differences(fname,traces,number=10,max_y=None):
	plt.clf()

	for trace in list(traces.values())[:number]:
		values = trace['u']
		magnitudes = list(map(np.linalg.norm,values))
		paired_magnitudes = list(zip(magnitudes,magnitudes[1:]))
		magnitude_differences = list(map(lambda pair: abs(pair[0] - pair[1]),paired_magnitudes))

		timesteps = [i for i in range(len(magnitude_differences))]

		plt.plot(timesteps, magnitude_differences)
		
	plt.xlabel("Timesteps")
	plt.ylabel("Control Deltas")
	# plt.axhline(0)

	if max_y:
		plt.ylim(bottom=0, top=max_y)
	else:
		plt.ylim(bottom=0, top=max(1,plt.ylim()[1]))

	# plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
	# plt.ylim(0, 1)

	# save the figure
	plt.savefig(f"{fname}.png",bbox_inches='tight')

	plt.clf()

# plot the traces of a simulation
def plot_trace(sim, model, args, stamp, partition, sim_values, sim_policy_inputs, filename="traces", show_plot = True, evaluation=None):
	plot_traces(args, stamp, model.plot_dimensions, partition, model, sim.results['traces'], line=False, num_traces=10, add_unsafe_box=False,filename=filename,show_plot=show_plot)
	heatmap(args, stamp, idx_show=model.plot_dimensions, slice_values=np.zeros(model.n), partition=partition, results=sim_values, filename="heatmap_satprob", show_plot=show_plot)
	heatmap(args, stamp, idx_show=model.plot_dimensions, slice_values=np.zeros(model.n), partition=partition, results=sim_policy_inputs[:,0], filename="heatmap_inputs", show_plot=show_plot)
	
	if evaluation and isinstance(evaluation, RL.Evaluate_Secondary.EnergyEfficiency):
		plot_action_magnitudes(fname=f'output/{args.model}/{filename}_{stamp}_action_magnitudes',traces=sim.results['traces'],max_y=np.linalg.norm(model.uMax))
	
	elif evaluation and isinstance(evaluation, RL.Evaluate_Secondary.ActionSmoothness):
		plot_action_magnitude_differences(fname=f'output/{args.model}/{filename}_{stamp}_action_deltas',traces=sim.results['traces'])
	
	if args.model == 'Pendulum':
		model.plot_trajectory_gif(np.array(sim.results['traces'][0]['x'])[:,0], filename=f'output/{args.model}/pendulum_{filename}_{stamp}.gif')

	if args.model == 'MountainCar':
		model.plot_trajectory_gif(np.array(sim.results['traces'][0]['x'])[:,0], filename=f'output/{args.model}/mountaincar_{filename}_{stamp}.gif')


def run_simulations(agent_envs, model, args, stamp, partition, policy, policy_inputs, sim_values, spheres, iterations=1000, verbose=False, show_plot=True, tracked_region=None):
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
	with open(f"output/{args.model}/{stamp}_results.txt", "a") as f:
		# write the sphere definition that we are using
		f.write(str(spheres))

		for (s,agent,vecnorm,evaluation) in agent_envs:
			f.write(f'Doing simulation for {s}\n')
			# run sims without RL agent
			sim = MonteCarloSim(model, partition, policy, policy_inputs, model.x0, project_action, verbose=verbose, iterations=iterations, evaluate_secondary=evaluation, tracked_region=tracked_region)
			f.write(f'Secondary Score: {sim.results["secondary_score"]}\n')
			f.write(f'Trace Average Secondary Score: {sim.results["secondary_score_average"]}\n')
			f.write(f'Empirical satisfaction probability: {sim.results['satprob']}\n')

			if isinstance(evaluation, RL.Evaluate_Secondary.DistanceToRegion):
				f.write(f'closest = {evaluation.closest}\n')
				evaluation.closest = float('inf')

			if tracked_region is not None:
				f.write(f'Times entered tracked region: {sim.results['tracked_region']}\n')

			f.write(f'Average trace length: {int(np.mean(list(map(lambda trace: len(trace['u']), list(sim.results['traces'].values())))))}\n')
			
			f.write('\n')

			# run sims with RL agent
			sim_rl = MonteCarloSim(model, partition, policy, policy_inputs, model.x0, project_action, verbose=verbose, iterations=iterations, evaluate_secondary=evaluation, agent=agent, spheres=spheres, vecnorm=vecnorm, tracked_region=tracked_region)
			f.write(f"Secondary Score with RL agent: {sim_rl.results['secondary_score']}\n")
			f.write(f"Trace Average Secondary Score with RL agent: {sim_rl.results['secondary_score_average']}\n")
			f.write(f"Empirical satisfaciton probability with RL agent: {sim_rl.results['satprob']}\n")
			if isinstance(evaluation, RL.Evaluate_Secondary.DistanceToRegion):
				f.write(f'closest = {evaluation.closest}\n')
				evaluation.closest = float('inf')
			
			if tracked_region is not None:
				f.write(f'Times RL agent entered tracked region: {sim_rl.results['tracked_region']}\n')


			f.write(f'Average trace length: {int(np.mean(list(map(lambda trace: len(trace['u']), list(sim_rl.results['traces'].values())))))}\n')

			f.write('\n\n')

			# plot the traces, heatmaps etc...
			plot_trace(
				sim=sim,
				model=model,
				args=args,
				stamp=stamp,
				partition=partition,
				sim_values=sim_values,
				sim_policy_inputs=policy_inputs,
				filename=s+'_NO_RL_WITH_SPHERE',
				show_plot=show_plot,
				evaluation=evaluation
			)

			plot_trace(
				sim=sim_rl,
				model=model,
				args=args,
				stamp=stamp,
				partition=partition,
				sim_values=sim_values,
				sim_policy_inputs=policy_inputs,
				filename=s+'_RL',
				show_plot=show_plot,
				evaluation=evaluation
			)


# TODO - double check this is right...
# find the shortest distance from a box to a box
def distance_from_box_to_box(box1_lower, box1_upper, box2_lower, box2_upper):
	sep = jnp.maximum(
        jnp.maximum(box1_lower - box2_upper, box2_lower - box1_upper),
        0.0
    )
	distance = jnp.linalg.norm(sep)
	return distance

# create a spatial box around the arena by producing 4 "thin" boxes around it
# spatial dimension of 2 => we expect first value of each coordinate to be x, and second to be y.
def create_borders(spatial_dimension,lower_bounds,upper_bounds):
	if spatial_dimension != 2:
		# TODO: support 3 and more dimensions...
		raise NotImplemented

	lower_x, lower_y = lower_bounds[0:1],lower_bounds[1:2]
	upper_x,upper_y = upper_bounds[0:1],upper_bounds[1:2]

	upper_border = np.array([
		np.concatenate([lower_x,upper_y,upper_bounds[2:]]),
		np.concatenate([upper_x,upper_y,upper_bounds[2:]])
	])
	lower_border = np.array([
		np.concatenate([lower_x,lower_y,lower_bounds[2:]]),
		np.concatenate([upper_x,lower_y,lower_bounds[2:]])
	])
	left_border = np.array([
		np.concatenate([lower_x,lower_y,lower_bounds[2:]]),
		np.concatenate([lower_x,upper_y,upper_bounds[2:]])
	])
	right_border = np.array([
		np.concatenate([upper_x,lower_y,lower_bounds[2:]]),
		np.concatenate([upper_x,upper_y,upper_bounds[2:]])
	])

	borders = [upper_border,lower_border,left_border,right_border]
	return borders