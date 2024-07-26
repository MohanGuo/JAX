import jax.numpy as jnp
import jax
import torch

def initialize_jax(N, M, scale):
    key = jax.random.PRNGKey(77)
    init_fn = jax.nn.initializers.variance_scaling(scale, "fan_avg", "uniform")
    return init_fn(key, (N, M))

def initialize_torch(N, M, gain, seed):
    torch.manual_seed(seed)
    layer = torch.nn.Linear(M, M)
    torch.nn.init.xavier_uniform_(layer.weight, gain=gain)
    return layer.weight.data


N, M = 100, 100

# jax_matrix = initialize_jax(N, M, 0.0001)
# print("JAX Matrix Mean:", jnp.mean(jax_matrix))
# print("JAX Matrix Variance:", jnp.var(jax_matrix))

jax_matrix = initialize_jax(N, M, 0.000001)
print("JAX Matrix Mean:", jnp.mean(jax_matrix))
print("JAX Matrix Variance:", jnp.var(jax_matrix))


torch_matrix = initialize_torch(N, M, 0.001, 0)
print("PyTorch Matrix Mean:", torch_matrix.mean())
print("PyTorch Matrix Variance:", torch_matrix.var())

# torch_matrix = initialize_torch(N, M, 0.0001, 87)
# print("PyTorch Matrix Mean:", torch_matrix.mean())
# print("PyTorch Matrix Variance:", torch_matrix.var())
