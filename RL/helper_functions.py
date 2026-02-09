# given an action proposed in the hyperrectangle [(-1,...,-1), (1,...,1)], find the corresponding real concrete action by scaling appropriately
def project_action(action, action_lower, action_upper):
	result = action_lower + (action + 1) * (action_upper - action_lower) / 2
	return result