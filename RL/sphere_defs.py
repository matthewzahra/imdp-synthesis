import jax.numpy as jnp
import numpy as np

'''
used to define different sphere specs so that we can test different satisfaction probabilities in one go
'''

class SphereDef:
	def __init__(self,thresholds,radii):
		assert thresholds.shape[0] == radii.shape[0]
		self.thresholds = thresholds
		self.radii = radii


def build_sphere_model(model_name: str) -> SphereDef:
	spheres = dict()
	spheres['Dubins_small'] = SphereDef(
		thresholds = jnp.array([3.4,3.2,3,2.8,2.6,2.4,2,1.6,1.2,0.8,0.4,0]),
        radii = jnp.array([
            [jnp.pi*0.34, 0.34],
            [jnp.pi*0.32, 0.32],
            [jnp.pi*0.3, 0.3],
            [jnp.pi*0.28, 0.28],
            [jnp.pi*0.26, 0.26],
            [jnp.pi*0.24, 0.24],
            [jnp.pi*0.20, 0.20],
            [jnp.pi*0.16, 0.16],
            [jnp.pi*0.12, 0.12],
            [jnp.pi*0.08, 0.08],
            [jnp.pi*0.04, 0.04],
            [0, 0],
        ])
	)

	# NOTE - there are no critical regions... just a goal region and then the sides of the arena
	spheres['MountainCar'] = SphereDef(
		thresholds=jnp.array([0]),
		radii=jnp.array([
			[0.2]
		])
	)

	return spheres[model_name]


def build_sphere_defs_test():
	# TODO - currently, we are NOT model specific here... we should be to be more general, like how we are in the function 'build_sphere_model'
	spheres = [
		SphereDef(
			thresholds = jnp.array([0]),
			radii = jnp.array([
				[0.15]
			])
		),

		SphereDef(
			thresholds = jnp.array([0]),
			radii = jnp.array([
				[0.2]
			])
		),

		SphereDef(
			thresholds = jnp.array([0]),
			radii = jnp.array([
				[0.25]
			])
		),


		SphereDef(
			thresholds = jnp.array([0]),
			radii = jnp.array([
				[0.3]
			])
		),


		# good > 82
		# SphereDef(
		# 	thresholds = jnp.array([2,1.8,1.6,1.4,1.2,1,0.8,0.6,0]),
		# 	radii = jnp.array([
		# 		[jnp.pi*0.24, 0],
		# 		[jnp.pi*0.23, 0],
		# 		[jnp.pi*0.22, 0],
		# 		[jnp.pi*0.21, 0],
		# 		[jnp.pi*0.2, 0],
		# 		[jnp.pi*0.18, 0],
		# 		[jnp.pi*0.16, 0],
		# 		[jnp.pi*0.14, 0],
		# 		[0,0]
		# 	])
		# ),

		# SphereDef(
		# 	thresholds = jnp.array([4,2,1.8,1.6,1.4,1.2,1,0.8,0.6,0]),
		# 	radii = jnp.array([
		# 		[jnp.pi*0.4, 0],
		# 		[jnp.pi*0.24, 0],
		# 		[jnp.pi*0.22, 0],
		# 		[jnp.pi*0.20, 0],
		# 		[jnp.pi*0.18, 0],
		# 		[jnp.pi*0.16, 0],
		# 		[jnp.pi*0.14, 0],
		# 		[jnp.pi*0.12, 0],
		# 		[jnp.pi*0.10, 0],
		# 		[0,0]
		# 	])
		# ),

		# SphereDef(
		# 	thresholds = jnp.array([3.6,3.4,3.2,3,2.8,2.6,2.4,2,1.6,1.2,0.8,0.4,0]),
		# 	radii = jnp.array([
		# 		[jnp.pi*0.36, 0.3],
		# 		[jnp.pi*0.34, 0.3],
		# 		[jnp.pi*0.32, 0.3],
		# 		[jnp.pi*0.3, 0.3],
		# 		[jnp.pi*0.28, 0.28],
		# 		[jnp.pi*0.26, 0.26],
		# 		[jnp.pi*0.24, 0.24],
		# 		[jnp.pi*0.20, 0.20],
		# 		[jnp.pi*0.16, 0.16],
		# 		[jnp.pi*0.12, 0.12],
		# 		[jnp.pi*0.08, 0.08],
		# 		[jnp.pi*0.04, 0.04],
		# 		[0, 0],
		# 	])
		# ),

	]

	return spheres

'''
geneate the values to clip and wrap at for each model
'''

def generate_clip_wrap_vals(model_name):
	vals_to_clip = {
		'Dubins_small':[[-np.pi*0.5,np.pi*0.5],[-3,3]],
		'MountainCar': [[-1,1]]
	}

	vals_to_wrap = {
		'Dubins_small':[None,None],
		'MountainCar': [None]
	}

	if model_name not in vals_to_clip.keys() or model_name not in vals_to_wrap.keys():
		raise ValueError(f"Values to clip/wrap not defined for model: {model_name}")

	return vals_to_clip[model_name], vals_to_wrap[model_name]
