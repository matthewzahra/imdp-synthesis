# TODO - double check this is right...
# find the shortest distance from a box to a box
def distance_from_box_to_box(box1_lower, box1_upper, box2_lower, box2_upper):
	sep = jnp.maximum(
        jnp.maximum(box1_lower - box2_upper, box2_lower - box1_upper),
        0.0
    )
	distance = jnp.linalg.norm(sep)
	return distance