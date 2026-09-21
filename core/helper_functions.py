import numpy as np


# assign each cell centerto its closest goal center
def reference_generator(centers, goal_centers, STATES_NUMBER):
    # Closest target cells computations
    # Number of partition cells 
    Ns = centers.shape[0]

    target_cell = np.zeros((Ns,STATES_NUMBER))

    for i in range(Ns):
        array_difference = goal_centers - centers[i,:]
        distances = np.zeros(len(goal_centers))
        for j in range(len(goal_centers)):
            distances[j] = np.linalg.norm(array_difference[j,:],1)
        min_idx = np.argmin(distances)
        target_cell[i,:] = goal_centers[min_idx,:].copy()

    return target_cell

def get_cell_index(centers,state):

    distances = np.sum((centers - state)**2, axis=1)
    return np.argmin(distances)