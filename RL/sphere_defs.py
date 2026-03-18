import jax.numpy as jnp

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

	# NOTE - there are no critical regions... So for dynamic spheres, just 1 threshold should be needed
	spheres['MountainCar'] = SphereDef(
		thresholds=jnp.array([0]),
		radii=jnp.array([0.2])
	)

	return spheres[model_name]


def build_sphere_defs_test():
	# TODO - currently, we are NOT model specific here... we should be to be more general, like how we are in the function 'build_sphere_model'
	spheres = [

		# seems decent...
		# SphereDef(
		# 	thresholds = jnp.array([2,1.8,1.6,1.4,1.2,1,0]),
		# 	radii = jnp.array([
		# 		[jnp.pi*0.24, 0],
		# 		[jnp.pi*0.22, 0],
		# 		[jnp.pi*0.2, 0],
		# 		[jnp.pi*0.18, 0],
		# 		[jnp.pi*0.16, 0],
		# 		[jnp.pi*0.14, 0],
		# 		[0,0]
		# 	])
		# ),

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
		# 	thresholds = jnp.array([0]),
		# 	radii = jnp.array([
		# 		[jnp.pi*0.25,0.25]
		# 	])
		# ),

		SphereDef(
			thresholds = jnp.array([3.6,3.4,3.2,3,2.8,2.6,2.4,2,1.6,1.2,0.8,0.4,0]),
			radii = jnp.array([
				[jnp.pi*0.36, 0.3],
				[jnp.pi*0.34, 0.3],
				[jnp.pi*0.32, 0.3],
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
		),

		# > 0.32
		# SphereDef(
		# 	thresholds = jnp.array([2,1.9,1.8,1.7,1.6,1.5,1.4,1.2,1,0.8,0]),
		# 	radii = jnp.array([
		# 		[jnp.pi*0.26, 0],
		# 		[jnp.pi*0.24, 0],
		# 		[jnp.pi*0.22, 0],
		# 		[jnp.pi*0.21, 0],
		# 		[jnp.pi*0.2, 0],
		# 		[jnp.pi*0.19, 0],
		# 		[jnp.pi*0.18, 0],
		# 		[jnp.pi*0.16, 0],
		# 		[jnp.pi*0.14, 0],
		# 		[jnp.pi*0.12, 0],
		# 		[0,0]
		# 	])
		# )

		# performs badly, 0.000166
		# SphereDef(
		# 	thresholds = jnp.array([2,1.8,1.6,1.4,1.2,1,0.8,0.6,0]),
		# 	radii = jnp.array([
		# 		[jnp.pi*0.24, 0.24],
		# 		[jnp.pi*0.23, 0.23],
		# 		[jnp.pi*0.22, 0.22],
		# 		[jnp.pi*0.21, 0.21],
		# 		[jnp.pi*0.2, 0.2],
		# 		[jnp.pi*0.18, 0.18],
		# 		[jnp.pi*0.16, 0.16],
		# 		[jnp.pi*0.14, 0.14],
		# 		[0,0]
		# 	])
		# ),

		# good, > 88 but can't turn...
		# SphereDef(
		# 	thresholds = jnp.array([2,1.8,1.6,1.4,1.2,1,0.8,0.6,0]),
		# 	radii = jnp.array([
		# 		[0, 0.24],
		# 		[0, 0.23],
		# 		[0, 0.22],
		# 		[0, 0.21],
		# 		[0, 0.2],
		# 		[0, 0.18],
		# 		[0, 0.16],
		# 		[0, 0.14],
		# 		[0,0]
		# 	])
		# ),

		# performs badly, 0.000326
		# SphereDef(
		# 	thresholds = jnp.array([2,1.8,1.6,1.4,1.2,1,0.8,0.6,0]),
		# 	radii = jnp.array([
		# 		[jnp.pi*0.24, 0.2],
		# 		[jnp.pi*0.23, 0.16],
		# 		[jnp.pi*0.22, 0.12],
		# 		[jnp.pi*0.21, 0.08],
		# 		[jnp.pi*0.2, 0.04],
		# 		[jnp.pi*0.18, 0.02],
		# 		[jnp.pi*0.16, 0],
		# 		[jnp.pi*0.14, 0],
		# 		[0,0]
		# 	])
		# ),

		# performs poorly, 0.004243
		# SphereDef(
		# 	thresholds = jnp.array([2,1.8,1.6,1.4,1.2,1,0.8,0.6,0]),
		# 	radii = jnp.array([
		# 		[jnp.pi*0.28, 0],
		# 		[jnp.pi*0.26, 0],
		# 		[jnp.pi*0.24, 0],
		# 		[jnp.pi*0.22, 0],
		# 		[jnp.pi*0.21, 0],
		# 		[jnp.pi*0.2, 0],
		# 		[jnp.pi*0.18, 0],
		# 		[jnp.pi*0.16, 0],
		# 		[0,0]
		# 	])
		# ),

		# SphereDef(
		# 	thresholds=jnp.array([5,3,2,1,0]),
		# 	radii=jnp.array([
		# 		[jnp.pi*0.4, 4],
		# 		[jnp.pi*0.3, 3],
		# 		[jnp.pi*0.2, 2],
		# 		[jnp.pi*0.1, 1],
		# 		[0,0]
		# 	])
		# ),


		# SphereDef(
		# 	thresholds=jnp.array([4,2,0]),
		# 	radii=jnp.array([
		# 		[jnp.pi*0.3, 3],
		# 		[jnp.pi*0.2, 2],
		# 		[0, 0],
		# 	])
		# ),


		# SphereDef(
		# 	thresholds=jnp.array([4,3,2,1,0]),
		# 	radii=jnp.array([
		# 		[jnp.pi*0.3, 0],
		# 		[jnp.pi*0.25, 0],
		# 		[jnp.pi*0.2, 0],
		# 		[jnp.pi*0.1, 0],
		# 		[0, 0],
		# 	])
		# ),
	]

	return spheres