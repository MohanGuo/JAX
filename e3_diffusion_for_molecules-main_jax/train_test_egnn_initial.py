import jax
import jax.numpy as jnp
from flax import linen as nn
from scipy.spatial.transform import Rotation as R
from egnn import models_jax
from egnn import models
# import egnn.models_jax# import EGNN_dynamics_QM9
# import egnn.models# import EGNN_dynamics_QM9
import torch
from jax import random

def compare_parameters(params_jax, model_torch):
    print("Comparing model parameters:")
    torch_params = [p.detach().numpy() for p in model_torch.parameters()]
    jax_params = [p for _, p in jax.tree_flatten(params_jax)[0]]

    for jax_param, torch_param in zip(jax_params, torch_params):
        print("JAX param:", jax_param.shape, "PyTorch param:", torch_param.shape)
        print("Are they close?", np.allclose(jax_param, torch_param, atol=1e-6))

def print_jax_params(params):
    print("JAX Model Parameters:")
    def print_params(subparams, path=''):
        # Check if the parameter is a dictionary and recurse
        if isinstance(subparams, dict):
            for k, v in subparams.items():
                print_params(v, path + k + '.')
        else:
            # Base case: print parameter name and values
            print(f"{path[:-1]}: shape={subparams.shape}")
            print(subparams)

    print_params(params)

def print_pytorch_params(model):
    print("PyTorch Model Parameters:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}")
        print(param.detach().numpy())

import os

def write_jax_params_to_file(params, filepath="/home/scur0396/JAX/run/outputs/jax_model_parameters.txt"):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, "w") as file:
        file.write("JAX Model Parameters:\n")
        def write_params(subparams, path=''):
            if isinstance(subparams, dict):
                for k, v in subparams.items():
                    write_params(v, path + k + '.')
            else:
                file.write(f"{path[:-1]}: shape={subparams.shape}\n")
                file.write(f"{subparams}\n")
                
        write_params(params)

def write_pytorch_params_to_file(model, filepath="/home/scur0396/JAX/run/outputs/pytorch_model_parameters.txt"):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, "w") as file:
        file.write("PyTorch Model Parameters:\n")
        for name, param in model.named_parameters():
            file.write(f"{name}: {param.shape}\n")
            file.write(f"{param.detach().numpy()}\n")


####################################### Compare Initialization ################################################
# Initialize the model
model_jax = models_jax.EGNN_dynamics_QM9(
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
context = jnp.ones((batch_size, n_nodes, model_jax.context_node_nf))  # Simple constant context
t = jnp.array([0.1])  # Constant time for simplicity

# Initialize model parameters of JAX
seed = 42
key = random.PRNGKey(seed)
jax_params = model_jax.init(key, t, xh, node_mask, edge_mask, context)


## Initialize model parameters of Pytorch
torch.manual_seed(seed)

model_pytorch = models.EGNN_dynamics_QM9(
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

write_jax_params_to_file(jax_params)
write_pytorch_params_to_file(model_pytorch)
####################################### Compare Initialization ################################################

# rotated_xh = xh.at[:, :, :model.n_dims].set(rotate_coordinates(xh[:, :, :model.n_dims], angle))
# rotated_xh = jnp.array([[[0.0, 0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 2.0, 1.0, 1.0], [0.0, 0.0, 3.0, 1.0, 1.0]]])
rotated_xh = jnp.array([[[-3.0, 0.0, 0.0, 1.0, 3.0], [0.0, 0.0, 0.0, 1.0, 0.0], [3.0, 0.0, 0.0, 2.0, 1.0]]])# Perform a forward pass through the model on the original and rotated inputs


original_outputs_jax = model_jax.apply(jax_params, t, xh, node_mask, edge_mask, context)
rotated_outputs_jax = model_jax.apply(jax_params, t, rotated_xh, node_mask, edge_mask, context)

# Forward pass in PyTorch
with torch.no_grad():
    xh_torch = torch.tensor(jax.device_get(xh), dtype=torch.float32)
    node_mask_torch = torch.tensor(jax.device_get(node_mask), dtype=torch.float32)
    edge_mask_torch = torch.tensor(jax.device_get(edge_mask), dtype=torch.float32)
    context_torch = torch.tensor(jax.device_get(context), dtype=torch.float32)
    t_numpy = jax.device_get(t)  # Convert from JAX to NumPy
    t_torch = torch.tensor(t_numpy, dtype=torch.float32)

    # Similarly, for the rotated_xh array
    rotated_xh_torch = torch.tensor(jax.device_get(rotated_xh), dtype=torch.float32)

    original_outputs_torch = model_pytorch._forward(t_torch, xh_torch, node_mask_torch, edge_mask_torch, context_torch)
    rotated_outputs_torch = model_pytorch._forward(t_torch, rotated_xh_torch, node_mask_torch, edge_mask_torch, context_torch)

# Print and compare outputs
print("JAX Original Outputs:", original_outputs_jax)
print("PyTorch Original Outputs:", original_outputs_torch)
print("JAX Rotated Outputs:", rotated_outputs_jax)
print("PyTorch Rotated Outputs:", rotated_outputs_torch)

# Print the outputs to compare
print(f"Original input: {xh}")
print(f"Rotated input: {rotated_xh}")
# print("Original Outputs:")
# print(original_outputs)
# print("Rotated Outputs:")
# print(rotated_outputs)

# Calculate the difference
# difference = jnp.abs(original_outputs - rotated_outputs)
# print("Difference between original and rotated outputs:")
# print(difference)
