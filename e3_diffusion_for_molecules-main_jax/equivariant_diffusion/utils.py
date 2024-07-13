import torch
import numpy as np
import jax.numpy as jnp
import jax


# class EMA():
#     def __init__(self, beta):
#         super().__init__()
#         self.beta = beta

#     def update_model_average(self, ma_model, current_model):
#         for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
#             old_weight, up_weight = ma_params.data, current_params.data
#             ma_params.data = self.update_average(old_weight, up_weight)

#     def update_average(self, old, new):
#         if old is None:
#             return new
#         return old * self.beta + (1 - self.beta) * new
class EMAState:
    def __init__(self, params, beta):
        self.ema = EMA(beta)
        self.params = params

class EMA:
    def __init__(self, beta):
        self.beta = beta

    def update_model_average(self, ma_params, current_params):
        return jax.tree.map(self.update_average, ma_params, current_params)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def sum_except_batch(x):
    return x.reshape(x.shape[0], -1).sum(axis=-1)


def remove_mean(x):
    mean = jnp.mean(x, axis=1, keepdims=True)
    x = x - mean
    return x



# (jnp.abs((x * (1 - node_mask)))).sum()
#so, node_mask=1 means the item should be masked
#the hypothesis is, if we pad with 

def remove_mean_with_mask(x, node_mask):
    masked_max_abs_value = (jnp.abs((x * (1 - node_mask)))).sum()
    #assert masked_max_abs_value < 1e-5, f'Error {masked_max_abs_value} too high'
    N = jnp.sum(node_mask,1, keepdims=True)
    mean = jnp.sum(x, 1, keepdims=True) / N
    # print(f"x in remove_mean_with_mask: {x}")
    # print(f"masked_max_abs_value: {masked_max_abs_value}")
    # print(f"N: {N}")
    # print(f"mean: {mean}")
    x = x - mean * node_mask
    return x

def remove_mean_with_mask_torch(x, node_mask):
    masked_max_abs_value = (x * (1 - node_mask)).abs().sum().item()
    assert masked_max_abs_value < 1e-5, f'Error {masked_max_abs_value} too high'
    N = node_mask.sum(1, keepdims=True)

    mean = torch.sum(x, dim=1, keepdim=True) / N
    x = x - mean * node_mask
    return x

def assert_mean_zero(x):
    mean = jnp.mean(x, axis=1, keepdims=True)
    assert jnp.abs(mean).max().item() < 1e-4


def assert_mean_zero_with_mask(x, node_mask, eps=1e-10):
    assert_correctly_masked(x, node_mask)
    largest_value = jnp.abs(x).max().item()
    error = jnp.abs(jnp.sum(x, axis=1, keepdims=True)).max().item()
    rel_error = error / (largest_value + eps)
    assert rel_error < 1e-2, f'Mean is not zero, relative_error {rel_error}'

def assert_mean_zero_with_mask_torch(x, node_mask, eps=1e-10):
    assert_correctly_masked_torch(x, node_mask)
    largest_value = x.abs().max().item()
    error = torch.sum(x, dim=1, keepdim=True).abs().max().item()
    rel_error = error / (largest_value + eps)
    assert rel_error < 1e-2, f'Mean is not zero, relative_error {rel_error}'


def assert_correctly_masked(variable, node_mask):
    assert jnp.abs((variable * (1 - node_mask))).max().item() < 1e-4, \
        'Variables not masked properly.'

def assert_correctly_masked_torch(variable, node_mask):
    assert (variable * (1 - node_mask)).abs().max().item() < 1e-4, \
        'Variables not masked properly.'

def center_gravity_zero_gaussian_log_likelihood(x):
    #assert x.ndim == 3
    B, N, D = x.shape
    assert_mean_zero(x)

    # r is invariant to a basis change in the relevant hyperplane.
    r2 = sum_except_batch(jax.lax.pow(x,2))

    # The relevant hyperplane is (N-1) * D dimensional.
    degrees_of_freedom = (N-1) * D

    # Normalizing constant and logpx are computed:
    log_normalizing_constant = -0.5 * degrees_of_freedom * np.log(2*np.pi)
    log_px = -0.5 * r2 + log_normalizing_constant

    return log_px


def sample_center_gravity_zero_gaussian(rng, size):
    #assert len(size) == 3
    # x = jnp.randn(size, device=device)
    x = jax.random.normal(rng, size)

    # This projection only works because Gaussian is rotation invariant around
    # zero and samples are independent!
    x_projected = remove_mean(x)
    return x_projected


def center_gravity_zero_gaussian_log_likelihood_with_mask(x, node_mask):
    #assert len(x.size()) == 3
    B, N_embedded, D = x.size()
    assert_mean_zero_with_mask(x, node_mask)

    # r is invariant to a basis change in the relevant hyperplane, the masked
    # out values will have zero contribution.
    r2 = sum_except_batch(jax.lax.pow(x,2))

    # The relevant hyperplane is (N-1) * D dimensional.
    N = node_mask.squeeze(2).sum(1)  # N has shape [B]
    degrees_of_freedom = (N-1) * D

    # Normalizing constant and logpx are computed:
    log_normalizing_constant = -0.5 * degrees_of_freedom * np.log(2*np.pi)
    log_px = -0.5 * r2 + log_normalizing_constant

    return log_px


def sample_center_gravity_zero_gaussian_with_mask(rng, size, node_mask):
    # print(f"size: {size}")
    assert len(size) == 3
    #TODO test
    x = jax.random.normal(rng, size)
    # x = jnp.full(size, 7)
    x_masked = x * node_mask
    # print(f"x in sample_center_gravity_zero_gaussian_with_mask: {x}, {x.shape}")
    # print(f"node_mask in sample_center_gravity_zero_gaussian_with_mask: {node_mask}, {node_mask.shape}")

    # This projection only works because Gaussian is rotation invariant around
    # zero and samples are independent!
    x_projected = remove_mean_with_mask(x_masked, node_mask)
    return x_projected


def standard_gaussian_log_likelihood(x):
    # Normalizing constant and logpx are computed:
    log_px = sum_except_batch(-0.5 * x * x - 0.5 * np.log(2*np.pi))
    return log_px


def sample_gaussian(rng, size):
    x = jax.random.normal(rng, size)
    # x = jnp.randn(size, device=device)
    return x


def standard_gaussian_log_likelihood_with_mask(x, node_mask):
    # Normalizing constant and logpx are computed:
    log_px_elementwise = -0.5 * x * x - 0.5 * np.log(2*np.pi)
    log_px = sum_except_batch(log_px_elementwise * node_mask)
    return log_px


def sample_gaussian_with_mask(rng, size, node_mask):
    #TODO test
    x = jax.random.normal(rng, size)
    # x = jnp.full(size, 4)
    x_masked = x * node_mask
    return x_masked
