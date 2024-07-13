import jax
import jax.numpy as jnp
from flax import linen as nn
from scipy.spatial.transform import Rotation as R
from egnn.models_jax import EGNN_dynamics_QM9
from equivariant_diffusion.luke_en_diffusion import EnVariationalDiffusion

# Define a simple EGNN dynamics model (ensure that this is appropriately defined in your project)
def create_simple_dynamics_model():
    return EGNN_dynamics_QM9(
        in_node_nf=3,  # Number of input node features
        context_node_nf=2,  # Number of context node features (assuming 2 for this example)
        n_dims=3,  # Dimensionality of spatial data
        hidden_nf=64,  # Number of hidden units
        n_layers=4,  # Number of layers
        attention=False,
        condition_time=True,
        tanh=False,
        mode='egnn_dynamics'
    )


# Initialize the EnVariationalDiffusion model
model = EnVariationalDiffusion(
    dynamics=create_simple_dynamics_model(),
    in_node_nf=5,  # Including spatial dimensions and features
    n_dims=3,
    timesteps=1000,
    parametrization='eps',
    noise_schedule='learned',
    noise_precision=1e-4,
    loss_type='vlb',
    norm_values=(1., 1., 1.),
    include_charges=True
)

# Define simple, specific inputs
batch_size = 2
n_nodes = 3
# Spatial coordinates in a small range and features as small integers
x = jnp.array([
    [[1.0, 0.0, 0.0, 0, 1], [0.0, 1.0, 0.0, 1, 0], [-1.0, 0.0, 0.0, 0, 1]],
    [[-1.0, 0.0, 0.0, 1, 0], [0.0, -1.0, 0.0, 0, 1], [1.0, 0.0, 0.0, 1, 0]]
])
h = {
    'categorical': jnp.array([
        [[1, 0], [0, 1], [1, 0]],
        [[0, 1], [1, 0], [0, 1]]
    ]),
    'integer': jnp.array([
        [[0], [1], [0]],
        [[1], [0], [1]]
    ])
}
node_mask = jnp.ones((batch_size, n_nodes, 1))
edge_mask = jnp.ones((batch_size, n_nodes, n_nodes, 1))
context = jnp.array([
    [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
    [[0.6, 0.5], [0.4, 0.3], [0.2, 0.1]]
])

# Initialize model parameters
params = model.init(jax.random.PRNGKey(0), jax.random.PRNGKey(1), x, h, node_mask, edge_mask, context, training=True)

# Simulate a forward pass
outputs = model.apply(params, jax.random.PRNGKey(2), x, h, node_mask, edge_mask, context, training=True)

# Print outputs
print("Outputs:", outputs)