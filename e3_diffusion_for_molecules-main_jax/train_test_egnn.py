import jax
import jax.numpy as jnp
from flax import linen as nn
from scipy.spatial.transform import Rotation as R
from egnn.models_jax import EGNN_dynamics_QM9

# Define a function to apply a 3D rotation to the spatial coordinates
def rotate_coordinates(x, angle, axis='z'):
    r = R.from_euler(axis, angle, degrees=True)
    rotation_matrix = jnp.array(r.as_matrix(), dtype=jnp.float32)
    return jnp.dot(x, rotation_matrix.T)

# Initialize the model
model = EGNN_dynamics_QM9(
    in_node_nf=3,
    context_node_nf=2,
    n_dims=3,
    hidden_nf=64,
    n_layers=4,
    attention=False,
    condition_time=True,
    tanh=False,
    mode='egnn_dynamics'
)

# Define sample batch size and number of nodes per batch
batch_size = 1
n_nodes = 3

# Create a simple input data: coordinates positioned along the x-axis with increasing values
xh = jnp.array([[[3.0, 0.0, 0.0, 2.0, 1.0], [0.0, 0.0, 0.0, 1.0, 0.0], [-3.0, 0.0, 0.0, 1.0, 3.0]]])
node_mask = jnp.ones((batch_size, n_nodes, 1))
edge_mask = jnp.ones((batch_size, n_nodes, n_nodes, 1))
context = jnp.ones((batch_size, n_nodes, model.context_node_nf))  # Simple constant context
t = jnp.array([0.1])  # Constant time for simplicity

# Initialize model parameters
params = model.init(jax.random.PRNGKey(2), t, xh, node_mask, edge_mask, context)

# Rotate the spatial part of xh by 90 degrees around the z-axis
# angle = 90
# rotated_xh = xh.at[:, :, :model.n_dims].set(rotate_coordinates(xh[:, :, :model.n_dims], angle))
# rotated_xh = jnp.array([[[0.0, 0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 2.0, 1.0, 1.0], [0.0, 0.0, 3.0, 1.0, 1.0]]])
rotated_xh = jnp.array([[[-3.0, 0.0, 0.0, 1.0, 3.0], [0.0, 0.0, 0.0, 1.0, 0.0], [3.0, 0.0, 0.0, 2.0, 1.0]]])# Perform a forward pass through the model on the original and rotated inputs
original_outputs = model.apply(params, t, xh, node_mask, edge_mask, context)
rotated_outputs = model.apply(params, t, rotated_xh, node_mask, edge_mask, context)

# Print the outputs to compare
print(f"Original input: {xh}")
print(f"Rotated input: {rotated_xh}")
print("Original Outputs:")
print(original_outputs)
print("Rotated Outputs:")
print(rotated_outputs)

# Calculate the difference
# difference = jnp.abs(original_outputs - rotated_outputs)
# print("Difference between original and rotated outputs:")
# print(difference)
