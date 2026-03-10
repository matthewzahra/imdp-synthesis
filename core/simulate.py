import numpy as np
from tqdm import tqdm
from typing import Optional
from RL.Evaluate_Secondary import EvaluateSecondary


class MonteCarloSim():
    '''
    Class to run Monte Carlo simulations on the discrete-time stochastic system closed under a fixed Markov policy.
    '''

    def __init__(self, model, partition, policy, policy_inputs, x0, project_action=None, iterations=100, sim_horizon=1000, random_initial_state=False, verbose=True, evaluate_secondary: Optional[EvaluateSecondary] = None, agent = None, spheres = None, vecnorm = None, tracked_region = None, **kwargs):
        '''
        tracked_region helps us to know if the simulations passed through a given box. it is of the form [lower,upper].
        NOTE - that it deals with CONCRETE STATES! 
        FURTHER NOTE: assumes that the tracked region is NOT a goal or a critical region.
        '''

        print('\nStarting Monte Carlo simulations...')

        self.verbose = verbose

        self.model = model
        self.partition = partition

        self.policy = policy
        self.policy_inputs = policy_inputs
        self.horizon = sim_horizon
        self.iterations = iterations
        self.random_initial_state = random_initial_state
        self.project_action = project_action

        self.evaluate_secondary = evaluate_secondary

        # Predefine noise to speed up computations
        self.define_noise_values()

        self.results = {
            'satprob': -1,
            'goal_reached': np.full(self.iterations, False, dtype=bool),
            'traces': {}, 
            'secondary_scores': np.zeros(self.iterations),
            'secondary_score': 0,    # secondary score (if we are using it)
            'tracked_region': np.zeros(self.iterations)
        }

        self.spheres = spheres
        self.agent = agent
        self.vecnorm = vecnorm
        self.tracked_region = tracked_region

        # For each of the monte carlo iterations
        for m in tqdm(range(self.iterations)):
            self.results['traces'][m], self.results['goal_reached'][m], self.results['secondary_scores'][m], self.results['tracked_region'][m] = self._runIteration(x0, m)

        self.results['satprob'] = np.mean(self.results['goal_reached'])
        self.results['secondary_score'] = np.mean(self.results['secondary_scores'])
        self.results['tracked_region'] = np.sum(self.results['tracked_region'])

    def define_noise_values(self):
        '''
        Predefine the noise values to speed up computations.
        '''

        # Gaussian noise mode
        self.noise = np.random.multivariate_normal(
            np.zeros(self.model.n), self.model.noise['cov'] ** 2,
            (self.iterations, self.horizon))
        
    def check_in_tracked_region(self,point):
        return np.all((point >= self.tracked_region[0]) & (point <= self.tracked_region[1]))


    def _runIteration(self, x0, m):
        '''
        Run a Monte Carlo simulation from x0.

        :param x0: Initial continuous state.
        :param m: Simulation number.
        :return:
            - trace: Dictionary containing the state and input at each time step.
            - success: Boolean indicating whether goal was reached.
        '''

        # Initialize variables at start of iteration
        success = False
        trace = {'k': [], 'x': [], 'u': []}
        k = 0

        # Initialize the current simulation
        x = np.zeros((self.horizon + 1, self.model.n))
        x_tuple = np.zeros((self.horizon + 1, self.model.n)).astype(int)
        s = np.zeros(self.horizon + 1).astype(int)
        u = np.zeros((self.horizon, self.model.p))
        a = np.zeros(self.horizon).astype(int)

        # Determine initial state
        if self.random_initial_state:
            s0, _ = self.partition.x2state(x0)
            x[0] = np.random.uniform(
                low=self.partition.regions['lower_bounds'][s0],
                high=self.partition.regions['lower_bounds'][s0])

        else:
            x[0] = x0

        # Add current state, belief, etc. to trace
        trace['k'] += [0]
        trace['x'] += [x[0]]

        # used to track if we enter the tracked region (if we are using it)
        tracked_region = 0

        ######

        # record the current secondary score - we will only use this if evaluate_secondary was set in the instantiation of this class
        current_secondary_score = 0

        # For each time step in the finite time horizon
        while k <= self.horizon:

            # Determine to which region the state belongs
            s[k], in_partition = self.partition.x2state(x[k])

            if in_partition:
                # Save that state is currently in state s_current
                x_tuple[k] = self.partition.region_idx_inv[s[k]]

            else:
                # Absorbing region reached
                x_tuple[k] = -1

                if self.verbose:
                    print(f'- Absorbing state reached at k = {k} (x = {x[k]}), so abort')
                return trace, success, current_secondary_score / k+1, tracked_region

            # If current region is the goal state ...
            if s[k] in self.partition.goal['idxs']:
                # Then abort the current iteration, as we have achieved the goal
                success = True
                if self.verbose:
                    print(f'- Goal state reached (x = {x[k]})')
                return trace, success, current_secondary_score / k+1, tracked_region

            # If current region is in critical states...
            elif s[k] in self.partition.critical['idxs']:
                # Then abort current iteration
                if self.verbose:
                    print('- Critical state reached, so abort')
                return trace, success, current_secondary_score / k+1, tracked_region

            # Check if we can still perform another action within the horizon
            elif k >= self.horizon:
                return trace, success, current_secondary_score / k+1, tracked_region

            # Retreive the action from the policy
            if len(self.policy.shape) == 1:
                # If infinite horizon, policy does not have a time index
                a[k] = self.policy[s[k]]

                concrete_state = x[k]

                if self.tracked_region is not None:
                    if self.check_in_tracked_region(concrete_state):
                        tracked_region = 1

                # check if we should be using an RL agent to make the decision 
                if self.agent is not None:

                    # need to add the policy action to the concrete state to get the observation
                    policy_action = self.policy_inputs[s[k]]
                    
                    # may need to also add the previous action
                    if self.evaluate_secondary.include_prev_action:
                        if k > 0:
                            prev_action = u[k-1]
                        else:
                            prev_action = u[0]
                        obs = np.concatenate([concrete_state,policy_action,prev_action])

                    else:
                        obs = np.concatenate([concrete_state,policy_action])

                    if self.vecnorm:
                        obs = self.vecnorm.normalize_obs(obs)
                    
                    # query the trained RL agent - the second item returned is hidden state - only needed for recurrent policies
                    proposed_action,_ = self.agent.predict(observation=obs, deterministic=True) # TODO - do we want this to be deterministic

                    # project the action
                    state_min,state_max = self.partition.regions['lower_bounds'][s[k]],self.partition.regions['lower_bounds'][s[k]]
                    action_set_lower_bounds, action_set_upper_bounds = self.spheres.get_action_sphere(action_centre=policy_action,state_min=state_min,state_max=state_max)
                    projected_action = self.project_action(proposed_action,action_set_lower_bounds,action_set_upper_bounds)

                    # save the action so that it can be executed later
                    u[k] = projected_action

                else:
                    u[k] = self.policy_inputs[s[k]]
            else:
                # If finite horizon, use action for the current time step k
                a[k] = self.policy[k, s[k]]
                u[k] = self.policy_inputs[k, s[k]]

            # if a[k] == -1:
            #     if self.verbose:
            #         print('No policy known, so abort')
            #     return trace, success

            ###

            # If loop was not aborted, we have a valid action
            if self.verbose:
                print(f'In state {s[k]} (x = {x[k]}), take action {a[k]} (u = {u[k]})')            
            x[k + 1] = self.model.step(x[k], u[k], self.noise[m, k])

            # if an evaluation was provided, make use of it
            if self.evaluate_secondary:
                current_secondary_score += self.evaluate_secondary.get_score(x[k],u[k])

            # Add current state, belief, etc. to trace
            trace['k'] += [k + 1]
            trace['u'] += [u[k]]
            trace['x'] += [x[k + 1]]

            # Increase iterator variable by one
            k += 1

        ######

        return trace, success, current_secondary_score / k, tracked_region
