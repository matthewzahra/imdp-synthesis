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
	# spheres['Dubins_small'] = SphereDef(
	# 	thresholds = jnp.array([3.4,3.2,3,2.8,2.6,2.4,2,1.6,1.2,0.8,0.4,0]),
    #     radii = jnp.array([
    #         [jnp.pi*0.34, 0.34],
    #         [jnp.pi*0.32, 0.32],
    #         [jnp.pi*0.3, 0.3],
    #         [jnp.pi*0.28, 0.28],
    #         [jnp.pi*0.26, 0.26],
    #         [jnp.pi*0.24, 0.24],
    #         [jnp.pi*0.20, 0.20],
    #         [jnp.pi*0.16, 0.16],
    #         [jnp.pi*0.12, 0.12],
    #         [jnp.pi*0.08, 0.08],
    #         [jnp.pi*0.04, 0.04],
    #         [0, 0],
    #     ])
	# )

	# spheres['Dubins_small'] = SphereDef(
	# 	thresholds = jnp.array([3.2,3,2.8,2.6,2.4,2,1.6,1.2,0.8,0.4,0]),
	# 	radii = jnp.array([
	# 		[jnp.pi*0.38, 0.18],
	# 		[jnp.pi*0.36, 0.16],
	# 		[jnp.pi*0.34, 0.16],
	# 		[jnp.pi*0.32, 0.15],
	# 		[jnp.pi*0.3, 0.15],
	# 		[jnp.pi*0.28, 0.14],
	# 		[jnp.pi*0.24, 0.13],
	# 		[jnp.pi*0.18, 0.12],
	# 		[jnp.pi*0.12, 0.10],
	# 		[jnp.pi*0.10, 0.10],
	# 		[0, 0],
	# 	])
	# )

	spheres['Dubins_small'] = SphereDef(
		thresholds = jnp.array([3.4,3.2,3,2.8,2.6,2.4,2,1.6,1.2,0.8,0.4,0.2,0]),
		radii = jnp.array([
			[jnp.pi*0.38, 0.18],
			[jnp.pi*0.36, 0.16],
			[jnp.pi*0.34, 0.16],
			[jnp.pi*0.32, 0.15],
			[jnp.pi*0.3, 0.15],
			[jnp.pi*0.28, 0.14],
			[jnp.pi*0.24, 0.13],
			[jnp.pi*0.18, 0.12],
			[jnp.pi*0.12, 0.10],
			[jnp.pi*0.10, 0.10],
			[jnp.pi*0.08, 0.10],
			[jnp.pi*0.06, 0.10],
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

	spheres['Drone2D'] = SphereDef(
			thresholds = jnp.array([3.4,3.2,3,2.8,2.6,2.4,2,1.6,1.2,0.8,0.4,0]),
			radii = jnp.array([
				[0.34, 0.34],
				[0.32, 0.32],
				[0.3, 0.3],
				[0.28, 0.28],
				[0.26, 0.26],
				[0.24, 0.24],
				[0.20, 0.20],
				[0.16, 0.16],
				[0.12, 0.12],
				[0.08, 0.08],
				[0.04, 0.04],
				[0, 0],
			])
		)
	
	spheres['Pendulum'] = SphereDef(
			thresholds=jnp.array([0]),
			radii=jnp.array([
				[0.25]
			])
		)

	return spheres[model_name]


def build_sphere_defs_test(model_name):
	spheres = dict()
	
	spheres["Dubins_small"] = [
		SphereDef(
				thresholds = jnp.array([3.3,3.1,2.9,2.7,2.5,2.1,1.7,1.3,0.9,0.5,0.3,0]),
				radii = jnp.array([
					[jnp.pi*0.38, 0.18],
					[jnp.pi*0.36, 0.16],
					[jnp.pi*0.34, 0.16],
					[jnp.pi*0.32, 0.15],
					[jnp.pi*0.3, 0.15],
					[jnp.pi*0.28, 0.14],
					[jnp.pi*0.24, 0.13],
					[jnp.pi*0.18, 0.12],
					[jnp.pi*0.14, 0.11],
					[jnp.pi*0.12, 0.10],
					[jnp.pi*0.10, 0.10],
					[0, 0],
				])
			),

		SphereDef(
			thresholds = jnp.array([3.4,3.2,3,2.8,2.6,2.4,2,1.6,1.2,0.8,0.4,0.2,0]),
			radii = jnp.array([
				[jnp.pi*0.38, 0.18],
				[jnp.pi*0.36, 0.16],
				[jnp.pi*0.34, 0.16],
				[jnp.pi*0.32, 0.15],
				[jnp.pi*0.3, 0.15],
				[jnp.pi*0.28, 0.14],
				[jnp.pi*0.24, 0.13],
				[jnp.pi*0.18, 0.12],
				[jnp.pi*0.12, 0.10],
				[jnp.pi*0.10, 0.10],
				[jnp.pi*0.08, 0.10],
				[jnp.pi*0.06, 0.10],
				[0, 0],
			])
		)
		# SphereDef(
		# 		thresholds = jnp.array([3.4,3.2,3,2.8,2.6,2.4,2,1.6,1.2,0.8,0.4,0]),
		# 		radii = jnp.array([
		# 			[jnp.pi*0.34, 0.34],
		# 			[jnp.pi*0.32, 0.32],
		# 			[jnp.pi*0.3, 0.3],
		# 			[jnp.pi*0.28, 0.28],
		# 			[jnp.pi*0.26, 0.26],
		# 			[jnp.pi*0.24, 0.24],
		# 			[jnp.pi*0.20, 0.20],
		# 			[jnp.pi*0.16, 0.16],
		# 			[jnp.pi*0.12, 0.12],
		# 			[jnp.pi*0.08, 0.08],
		# 			[jnp.pi*0.04, 0.04],
		# 			[0, 0],
		# 		])
		# 	),
		# SphereDef(
		# 		thresholds = jnp.array([3.6,3.4,3.2,3,2.8,2.6,2.4,2,1.6,1.2,0.8,0.4,0]),
		# 		radii = jnp.array([
		# 			[jnp.pi*0.36, 0.36],
		# 			[jnp.pi*0.34, 0.34],
		# 			[jnp.pi*0.32, 0.32],
		# 			[jnp.pi*0.3, 0.3],
		# 			[jnp.pi*0.28, 0.28],
		# 			[jnp.pi*0.26, 0.26],
		# 			[jnp.pi*0.24, 0.24],
		# 			[jnp.pi*0.20, 0.20],
		# 			[jnp.pi*0.16, 0.16],
		# 			[jnp.pi*0.12, 0.12],
		# 			[jnp.pi*0.08, 0.08],
		# 			[jnp.pi*0.04, 0.04],
		# 			[0, 0],
		# 		])
		# 	),
	]

	spheres["Drone2D"] = [
 		# SphereDef(
		# 	thresholds = jnp.array([3.4,3.2,3,2.8,2.6,2.4,2,1.6,1.2,0.8,0.4,0]),
		# 	radii = jnp.array([
		# 		[0.34, 0.34],
		# 		[0.32, 0.32],
		# 		[0.3, 0.3],
		# 		[0.28, 0.28],
		# 		[0.26, 0.26],
		# 		[0.24, 0.24],
		# 		[0.20, 0.20],
		# 		[0.16, 0.16],
		# 		[0.12, 0.12],
		# 		[0.08, 0.08],
		# 		[0.04, 0.04],
		# 		[0, 0],
		# 	])
		# ),
		SphereDef(
			thresholds = jnp.array([0]),
			radii = jnp.array([
				[0, 0],
			])
		)
	]

	# NOTE - there are no critical regions... just a goal region and then the sides of the arena
	spheres['Pendulum'] = [
		SphereDef(
			thresholds=jnp.array([0]),
			radii=jnp.array([
				[0.25]
			])
		),
		SphereDef(
			thresholds=jnp.array([0]),
			radii=jnp.array([
				[0.3]
			])
		),
		# SphereDef(
		# 	thresholds=jnp.array([0]),
		# 	radii=jnp.array([
		# 		[0.35]
		# 	])
		# ),
		# SphereDef(
		# 	thresholds=jnp.array([0]),
		# 	radii=jnp.array([
		# 		[0.4]
		# 	])
		# ),
		# SphereDef(
		# 	thresholds=jnp.array([0]),
		# 	radii=jnp.array([
		# 		[0.45]
		# 	])
		# ),
	
	]

	return spheres[model_name]

'''
geneate the values to clip and wrap at for each model
'''
def generate_clip_wrap_vals(model_name):
	vals_to_clip = {
		'Dubins_small':[[-np.pi*0.5,np.pi*0.5],[-3,3]],
		'MountainCar': [[-1,1]],
		'Drone2D': [[-3,3],[-3,3]],
		'Pendulum':[[-2,2]]
	}

	vals_to_wrap = {
		'Dubins_small':[None,None],
		'MountainCar': [None],
		'Drone2D': [None,None],
		'Pendulum': [None]
	}

	if model_name not in vals_to_clip.keys() or model_name not in vals_to_wrap.keys():
		raise ValueError(f"Values to clip/wrap not defined for model: {model_name}")

	return vals_to_clip[model_name], vals_to_wrap[model_name]
