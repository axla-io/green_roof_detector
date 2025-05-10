# Define a new Flax Module that implements a non-standard operation
# e.g., SpatialFeatureAmplifier: enhances edge/vegetation using NDVI-style logic
# Implement the layer as a function using `jax.numpy` and integrate it into Flax model
# Ensure it's differentiable and works with `grad`