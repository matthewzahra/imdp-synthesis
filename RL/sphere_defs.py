import jax.numpy as jnp

'''
used to define different sphere specs so that we can test different satisfaction probabilities in one go
'''

class SphereDef:
	def __init__(self,thresholds,radii):
		assert thresholds.shape[0] == radii.shape[0]
		self.thresholds = thresholds
		self.radii = radii

def build_sphere_defs():
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

		SphereDef(
			thresholds = jnp.array([4,2,1.8,1.6,1.4,1.2,1,0.8,0.6,0]),
			radii = jnp.array([
				[jnp.pi*0.4, 0],
				[jnp.pi*0.24, 0],
				[jnp.pi*0.22, 0],
				[jnp.pi*0.20, 0],
				[jnp.pi*0.18, 0],
				[jnp.pi*0.16, 0],
				[jnp.pi*0.14, 0],
				[jnp.pi*0.12, 0],
				[jnp.pi*0.10, 0],
				[0,0]
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