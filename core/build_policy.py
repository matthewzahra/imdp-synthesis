import jax.numpy as jnp
from RL.generate_action_sets import Spheres
from RL.helper_functions import create_borders
import numpy as np
from core.actions_forward import RectangularForward
from core.Gaussian_probabilities import compute_probability_intervals
from core.imdp import IMDP
import time
import jax

# build the policy, including performing value/policy iteration
def build_policy(model, partition, args, stamp, t, thresholds=None, radii_options=None, vals_to_clip=None, vals_to_wrap=None, continuous=False, radii_funcs=None, reinforcement_learning=False):
	# Create actions based on forward reachable sets
	if reinforcement_learning:
		# TODO - think about how better to construct the radii - these should take into account the magnitude of each component on the action space that we expect so that the spheres are of the approrpriate size

		# add 4 critical regions to contain the whole arena
		# NOTE: we assume that the first 2 values of each point are the physical x and y coordinates
		# TODO - currently we assume only a 2D space
		boundaries = model.partition['boundary']
		borders = create_borders(spatial_dimension=2,lower_bounds=boundaries[0],upper_bounds=boundaries[1])
		if model.critical.size == 0:
			critical_regions = np.concatenate([borders, model.goal])
		else:
			critical_regions = np.concatenate([model.critical, borders, model.goal])

		spheres = Spheres(
			thresholds=thresholds,
			radii_options=radii_options,
			vals_to_clip=vals_to_clip,
			vals_to_wrap=vals_to_wrap,
			critical_regions=critical_regions, # include the borders and goal region as critical regions
			model=model,
			radii_funcs=radii_funcs,
			continuous=continuous
		)

		actions = RectangularForward(args=args, partition=partition, model=model, action_spheres=spheres)     
		actions_inputs = actions.id_to_input   
	else:
		spheres = None
		actions = RectangularForward(args=args, partition=partition, model=model)
		actions_inputs = actions.id_to_input


	# TODO - edit this function to use the new probability intervals 

	P_full, S_id, A_id, P_absorbing = compute_probability_intervals(args=args, 
													model=model, 
													partition=partition, 
													actions=actions,
													vectorized=True)

	del actions

	imdp = IMDP(
		partition=partition,
		states=np.array(partition.regions['idxs']),
		actions_inputs=actions_inputs,
		x0=model.x0,
		goal_regions=np.array(partition.goal['bools']),
		critical_regions=np.array(partition.critical['bools']),
		P_full=P_full,
		S_id=S_id,
		A_id=A_id,
		P_absorbing=P_absorbing
	)

	print(f'- Generating abstraction took: {(time.time() - t):.3f} sec.')

	
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

		sat_prob = V[partition.x2state(model.x0)]
		with open(f"{stamp}_results.txt", "a") as f:
			f.write(f"Satisfaction probability: {sat_prob}\n\n")

	return V, policy, policy_inputs, spheres