from equivariant_diffusion import utils
import numpy as np
import math
# import torch
from egnn import models
# from torch.nn import functional as F
from equivariant_diffusion import utils as diffusion_utils

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.nn import softplus
from jax.nn.initializers import kaiming_uniform, uniform
import math
from jax.scipy.special import expit
from jax.scipy.stats import norm

# Defining some useful util functions.
def expm1(x: jnp.ndarray) -> jnp.ndarray:
    """exp(x) - 1"""
    return jnp.expm1(x)

def softplus(x: jnp.ndarray) -> jnp.ndarray:
    """Softplus function"""
    return jax.nn.softplus(x)

def sum_except_batch(x: jnp.ndarray) -> jnp.ndarray:
    """sum over all the dimensions except batch"""
    return x.reshape(x.shape[0], -1).sum(-1)

#If the ratio is too small, the noise change is too large.
def clip_noise_schedule(alphas2, clip_value=0.001):
    """
    For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1. This may help improve stability during
    sampling.

    Parameters:
        alphas2 (jnp.ndarray): Array of alpha^2 values representing the noise schedule.
        clip_value (float): The minimum value to which alpha_t / alpha_t-1 can be clipped.

    Returns:
        jnp.ndarray: The modified alphas2 array after applying the clipping and cumulative product.
    """
    # Ensure that alphas2 is a JAX array for compatibility
    alphas2 = jnp.array(alphas2)

    # Concatenate to introduce the initial value for stable division
    alphas2 = jnp.concatenate([jnp.ones(1), alphas2], axis=0)

    # Compute the ratio of successive alphas2, and clip each step
    alphas_step = (alphas2[1:] / alphas2[:-1])
    alphas_step = jnp.clip(alphas_step, a_min=clip_value, a_max=1.)

    # Compute the cumulative product to reconstruct the alphas2 sequence
    alphas2 = jnp.cumprod(alphas_step, axis=0)

    return alphas2


def polynomial_schedule(timesteps: int, s=1e-4, power=3.):
    """
    A noise schedule based on a simple polynomial equation: 1 - x^power.
    """
    steps = timesteps + 1
    x = jnp.linspace(0, steps, steps)
    alphas2 = (1 - jnp.power(x / steps, power))**2

    alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)

    precision = 1 - 2 * s

    alphas2 = precision * alphas2 + s

    return alphas2

#the intuition behind still needs to be investigated
def cosine_beta_schedule(timesteps, s=0.008, raise_to_power: float = 1):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 2
    x = jnp.linspace(0, steps, steps)
    alphas_cumprod = jnp.cos(((x / steps) + s) / (1 + s) * jnp.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = jnp.clip(betas, a_min=0, a_max=0.999)
    alphas = 1. - betas
    alphas_cumprod = jnp.cumprod(alphas, axis=0)

    if raise_to_power != 1:
        alphas_cumprod = jnp.power(alphas_cumprod, raise_to_power)

    return alphas_cumprod


def gaussian_entropy(mu, sigma):
    """
    Calculate the entropy of a Gaussian distribution for given mean and standard deviation.

    Parameters:
        mu (jnp.ndarray): The mean vector of the Gaussian distribution.
        sigma (jnp.ndarray): The standard deviation of the Gaussian distribution.

    Returns:
        jnp.ndarray: The entropy of the Gaussian distribution for each element in the batch.
    """
    zeros = jnp.zeros_like(mu)
    return sum_except_batch(
        zeros + 0.5 * jnp.log(2 * jnp.pi * sigma**2) + 0.5
    )


def gaussian_KL(q_mu, q_sigma, p_mu, p_sigma, node_mask):
    """
    Computes the KL divergence between two normal distributions.

    Args:
        q_mu (jnp.ndarray): Mean of distribution q.
        q_sigma (jnp.ndarray): Standard deviation of distribution q.
        p_mu (jnp.ndarray): Mean of distribution p.
        p_sigma (jnp.ndarray): Standard deviation of distribution p.
        node_mask (jnp.ndarray): Mask that selects the nodes to consider in the sum.

    Returns:
        jnp.ndarray: The KL divergence, summed over all dimensions except the batch dim.
    """
    kl_div = (
        jnp.log(p_sigma / q_sigma) +
        0.5 * (q_sigma**2 + (q_mu - p_mu)**2) / p_sigma**2 -
        0.5
    )
    # Applying the node mask and summing except for the batch dimension
    return sum_except_batch(kl_div * node_mask)


def gaussian_KL_for_dimension(q_mu, q_sigma, p_mu, p_sigma, d):
    """
    Computes the KL divergence between two normal distributions over a specific dimension d.

    Args:
        q_mu (jnp.ndarray): Mean of distribution q.
        q_sigma (jnp.ndarray): Standard deviation of distribution q.
        p_mu (jnp.ndarray): Mean of distribution p.
        p_sigma (jnp.ndarray): Standard deviation of distribution p.
        d (int): The dimension over which to compute the KL divergence.

    Returns:
        jnp.ndarray: The KL divergence, summed over all dimensions except the batch dim.
    """
    mu_norm2 = sum_except_batch((q_mu - p_mu)**2)
    # Ensure the standard deviations are 1-dimensional
    assert q_sigma.ndim == 1
    assert p_sigma.ndim == 1
    kl_div = (
        d * jnp.log(p_sigma / q_sigma) +
        0.5 * (d * q_sigma**2 + mu_norm2) / p_sigma**2 -
        0.5 * d
    )
    return kl_div


class PositiveLinear(nn.Module):
    """Linear layer with weights forced to be positive."""
    in_features: int
    out_features: int
    use_bias: bool = True
    weight_init_offset: float = -2.0

    def __call__(self, inputs):
        # weight initialization
        kaiming_init = kaiming_uniform()
        
        weight_shape = (self.out_features, self.in_features)
        weight = self.param('weight', kaiming_init, weight_shape)
        weight = softplus(weight + self.weight_init_offset)  # make sure the weights are positive

        # bias initialization
        if self.use_bias:
            # calculate the bound of bias
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            bias_shape = (self.out_features,)
            bias = self.param('bias', uniform(-bound, bound), bias_shape)
        else:
            bias = None

        # forward pass
        return jnp.dot(inputs, weight.T) + bias if self.use_bias else jnp.dot(inputs, weight.T)


## Position embedding encoding
class SinusoidalPosEmb(nn.Module):
    dim: int

    def setup(self):
        pass  

    def __call__(self, x):
        x = x.squeeze() * 1000
        assert x.ndim == 1  
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = jnp.concatenate((jnp.sin(emb), jnp.cos(emb)), axis=-1)
        return emb
    

class PredefinedNoiseSchedule():
    noise_schedule: str
    timesteps: int
    precision: float


    def __init__(self, noise_schedule, timesteps, noise_precision):
        self.noise_schedule=noise_schedule
        self.timesteps=timesteps
        self.precision=noise_precision

        if self.noise_schedule == 'cosine':
            alphas2 = cosine_beta_schedule(self.timesteps)
        elif 'polynomial' in self.noise_schedule:
            splits = self.noise_schedule.split('_')
            assert len(splits) == 2
            power = float(splits[1])
            alphas2 = polynomial_schedule(self.timesteps, s=self.precision, power=power)
        else:
            raise ValueError(self.noise_schedule)

        sigmas2 = 1 - alphas2
        log_alphas2 = jnp.log(alphas2)
        log_sigmas2 = jnp.log(sigmas2)
        log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2

        # Normally jax.nn.Parameter is not used for unlearnable parameters
        self.gamma = jnp.array(-log_alphas2_to_sigmas2, dtype=jnp.float32)

    def __call__(self, t):
        # t is in [0,1] and normalized
        t_int = jnp.round(t * self.timesteps).astype(jnp.int32)
        return self.gamma[t_int]


#TODO what does gamma control?
class GammaNetwork():
    # def setup(self):
    #     self.l1 = PositiveLinear(1, 1)
    #     self.l2 = PositiveLinear(1, 1024)
    #     self.l3 = PositiveLinear(1024, 1)

    #     # initialization
    #     self.gamma_0 = self.param('gamma_0', lambda rng, shape: -5 * jnp.ones(shape), ())
    #     self.gamma_1 = self.param('gamma_1', lambda rng, shape: 10 * jnp.ones(shape), ())

    #calculate the gamma at time t
    def gamma_tilde(self, t):
        l1_t = self.l1(t)
        return l1_t + self.l3(jax.nn.sigmoid(self.l2(l1_t)))
    
    def show_schedule(self, num_steps=50):
        t = jnp.linspace(0, 1, num_steps).reshape((num_steps, 1))
        gamma = self(t)  # __call__
        print('Gamma schedule:')
        print(gamma.reshape(num_steps))

    # @nn.compact 
    def __call__(self, t):
        # print("t type is ",type(t))
        # print("input size is ",t.shape)
        zeros, ones = jnp.zeros_like(t), jnp.ones_like(t)
        gamma_tilde_0 = self.gamma_tilde(zeros)
        gamma_tilde_1 = self.gamma_tilde(ones)
        gamma_tilde_t = self.gamma_tilde(t)

        # normalize gamma to [0,1]
        normalized_gamma = (gamma_tilde_t - gamma_tilde_0) / (gamma_tilde_1 - gamma_tilde_0)

        # rescale to [gamma_0, gamma_1]
        gamma = self.gamma_0 + (self.gamma_1 - self.gamma_0) * normalized_gamma
        return gamma


def cdf_standard_gaussian(x):
    return 0.5 * (1. + jax.scipy.special.erf(x / math.sqrt(2)))


class EnVariationalDiffusion():
    """
    The E(n) Diffusion Module.
    """

    dynamics: models.EGNN_dynamics_QM9
    in_node_nf: int
    n_dims: int
    timesteps: int = 1000
    parametrization: str = 'eps'
    noise_schedule: str = 'learned'
    noise_precision: float = 1e-4
    loss_type: str = 'vlb'
    norm_values: tuple = (1., 1., 1.)
    norm_biases: tuple = (None, 0., 0.)
    include_charges: bool = True


    def __init__(self,dynamics,
            in_node_nf,
            n_dims,
            timesteps,
            noise_schedule,
            noise_precision,
            loss_type,
            norm_values,
            include_charges):
        self.dynamics=dynamics
        self.in_node_nf=in_node_nf
        self.n_dims=n_dims
        self.timesteps=timesteps
        self.noise_schedule=noise_schedule
        self.noise_precision=noise_precision
        self.loss_type=loss_type
        self.norm_values=norm_values
        self.include_charges=include_charges
        
        

    def setup2(self):
        assert self.loss_type in {'vlb', 'l2'}
        if self.noise_schedule == 'learned':
            assert self.loss_type == 'vlb', 'A noise schedule can only be learned' \
                                       ' with a vlb objective.'

        
        # Only supported parametrization.
        assert self.parametrization == 'eps'

        if self.noise_schedule == 'learned':
            self.gamma = GammaNetwork()
        else:
            self.gamma = PredefinedNoiseSchedule(self.noise_schedule, self.timesteps, self.noise_precision)

        self.num_classes = self.in_node_nf - self.include_charges

        #register a variable without updating
        # self.buffer = self.variable('states', 'buffer', lambda: jnp.zeros(1))
        if self.noise_schedule != 'learned':
            self.check_issues_norm_values()
        self.training=True
        print("setting self.T here because I don't know where it should be set")
        self.T=self.timesteps


    # @nn.compact 
    def __call__(self, x, h, node_mask=None, edge_mask=None, context=None):
        """
        Computes the loss (type l2 or NLL) if training. And if eval then always computes NLL.
        """
        # Normalize data, take into account volume change in x.
        x, h, delta_log_px = self.normalize(x, h, node_mask)

        # Reset delta_log_px if not vlb objective.
        if self.training and self.loss_type == 'l2':
            delta_log_px = jnp.zeros_like(delta_log_px)

        if self.training:
            # Only 1 forward pass when t0_always is False.
            loss, loss_dict = self.compute_loss(x, h, node_mask, edge_mask, context, t0_always=False)
        else:
            # Less variance in the estimator, costs two forward passes.
            loss, loss_dict = self.compute_loss(x, h, node_mask, edge_mask, context, t0_always=True)

        neg_log_pxh = loss

        # Correct for normalization on x.
        assert neg_log_pxh.shape == delta_log_px.shape
        neg_log_pxh = neg_log_pxh - delta_log_px

        return neg_log_pxh
    

    def check_issues_norm_values(self, num_stdevs=8):
        zeros = jnp.zeros((1, 1))
        gamma_0 = self.gamma(zeros)
        sigma_0 = self.sigma(gamma_0, zeros)  # no need for .item()，JAX array is kept.

        # Checked if 1 / norm_value is still larger than 10 * standard
        # deviation.
        max_norm_value = max(self.norm_values[1], self.norm_values[2])

        if sigma_0 * num_stdevs > 1. / max_norm_value:
            raise ValueError(
                f'Value for normalization value {max_norm_value} probably too '
                f'large with sigma_0 {sigma_0[0]:.5f} and '
                f'1 / norm_value = {1. / max_norm_value}')

    #The forward of dynamics
    def phi(self, x, t, node_mask, edge_mask, context):
        # net_out shape is  torch.Size([64, 27, 8])
        net_out = self.dynamics._forward(t, x, node_mask, edge_mask, context)

        #  print(net_out.shape)
        return net_out
    

    def inflate_batch_array(self, array, target):
        """
        Inflates the batch array (array) with only a single axis (i.e. shape = (batch_size,), or possibly more empty
        axes (i.e. shape (batch_size, 1, ..., 1)) to match the target shape.
        """
        """
        In JAX, reshape the array to match the batch size of `target` with additional singleton dimensions
        to match the overall dimensions of `target`.
        """
        # Calculate the new shape: batch size of `array`, and the rest are singleton dimensions
        # to match the dimensionality of `target`.
        target_shape = (array.shape[0],) + (1,) * (len(target.shape) - 1)
        return array.reshape(target_shape)

    def sigma(self, gamma, target_tensor):
        """Computes sigma given gamma."""
        return self.inflate_batch_array(jnp.sqrt(jax.nn.sigmoid(gamma)), target_tensor)

    def alpha(self, gamma, target_tensor):
        """Computes alpha given gamma."""
        return self.inflate_batch_array(jnp.sqrt(jax.nn.sigmoid(-gamma)), target_tensor)

    def SNR(self, gamma):
        """Computes signal to noise ratio (alpha^2/sigma^2) given gamma."""
        return jnp.exp(-gamma)

    #TODO how to understand this?
    def subspace_dimensionality(self, node_mask):
        """Compute the dimensionality on translation-invariant linear subspace where distributions on x are defined."""
        number_of_nodes = jnp.sum(node_mask.squeeze(2), axis=1)
        return (number_of_nodes - 1) * self.n_dims

    #normalize features. how to give norm_biases and norm_values
    def normalize(self, x, h, node_mask):
        x = x / self.norm_values[0]
        delta_log_px = -self.subspace_dimensionality(node_mask) * jnp.log(self.norm_values[0]) #only position needs to reduce dimensionality?

        # Casting to float in case h still has long or int type.
        h_cat = (h['categorical'].astype(jnp.float32) - self.norm_biases[1]) / self.norm_values[1] * node_mask
        h_int = (h['integer'].astype(jnp.float32) - self.norm_biases[2]) / self.norm_values[2]

        if self.include_charges:
            h_int = h_int * node_mask

        # Create new h dictionary.
        h = {'categorical': h_cat, 'integer': h_int}

        return x, h, delta_log_px
    

    def unnormalize(self, x, h_cat, h_int, node_mask):
        x = x * self.norm_values[0] #position
        h_cat = h_cat * self.norm_values[1] + self.norm_biases[1] #categorical features
        h_cat = h_cat * node_mask #only categorical features have mask, which is the type of atoms.
        h_int = h_int * self.norm_values[2] + self.norm_biases[2] #integer features

        if self.include_charges:
            h_int = h_int * node_mask

        return x, h_cat, h_int

    def unnormalize_z(self, z, node_mask):
        # Parse from z
        x, h_cat = z[:, :, 0:self.n_dims], z[:, :, self.n_dims:self.n_dims+self.num_classes]
        h_int = z[:, :, self.n_dims+self.num_classes:self.n_dims+self.num_classes+1]
        assert h_int.shape[2] == (1 if self.include_charges else 0)

        # Unnormalize
        x, h_cat, h_int = self.unnormalize(x, h_cat, h_int, node_mask)
        output = jnp.concatenate([x, h_cat, h_int], axis=2)
        return output

    

    def sigma_and_alpha_t_given_s(self, gamma_t, gamma_s, target_tensor):
        """
        Computes sigma t given s, using gamma_t and gamma_s. Used during sampling.
        These are defined as:
            alpha t given s = alpha t / alpha s,
            sigma t given s = sqrt(1 - (alpha t given s) ^2 ).
        """
        sigma2_t_given_s = self.inflate_batch_array(
            -expm1(jnp.log1p(jnp.exp(gamma_s)) - jnp.log1p(jnp.exp(gamma_t))), target_tensor
        )

        # JAX  log-sigmoid (logistic sigmoid  log)
        log_alpha2_t = jnp.log(expit(-gamma_t))
        log_alpha2_s = jnp.log(expit(-gamma_s))
        log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s

        alpha_t_given_s = jnp.exp(0.5 * log_alpha2_t_given_s)
        alpha_t_given_s = self.inflate_batch_array(alpha_t_given_s, target_tensor)

        sigma_t_given_s = jnp.sqrt(sigma2_t_given_s)

        return sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s

    def kl_prior(self, xh, node_mask):
        """Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).

        This is essentially a lot of work for something that is in practice negligible in the loss. However, you
        compute it so that you see it when you've made a mistake in your noise schedule.
        """

        # Calculate the last alpha value, alpha_T.
        ones = jnp.ones((xh.shape[0], 1))
        gamma_T = self.gamma(ones) #calculate the output of gammanetwork of inputting all ones
        alpha_T = self.alpha(gamma_T, xh) #calculate the alpha given gamma

        # Compute means.
        mu_T = alpha_T * xh
        mu_T_x, mu_T_h = mu_T[:, :, :self.n_dims], mu_T[:, :, self.n_dims:] #split means of x and h

        # Compute standard deviations (only batch axis for x-part, inflated for h-part).
        sigma_T_x = self.sigma(gamma_T, mu_T_x).squeeze()  # TODO
        sigma_T_h = self.sigma(gamma_T, mu_T_h)

        # Compute KL for h-part.
        zeros_h, ones_h = jnp.zeros_like(mu_T_h), jnp.ones_like(sigma_T_h)
        kl_distance_h = gaussian_KL(mu_T_h, sigma_T_h, zeros_h, ones_h, node_mask)
        # kl_distance_h = gaussian_KL(mu_T_h, sigma_T_h, zeros, ones, node_mask)

        # Compute KL for x-part.
        zeros_x, ones_x = jnp.zeros_like(mu_T_x), jnp.ones_like(sigma_T_x)
        subspace_d = self.subspace_dimensionality(node_mask)
        kl_distance_x = gaussian_KL_for_dimension(mu_T_x, sigma_T_x, zeros_x, ones_x, d=subspace_d)

        return kl_distance_x + kl_distance_h

    def compute_x_pred(self, net_out, zt, gamma_t):
        """Commputes x_pred, i.e. the most likely prediction of x."""
        if self.parametrization == 'x':
            x_pred = net_out
        elif self.parametrization == 'eps':
            sigma_t = self.sigma(gamma_t, target_tensor=net_out)
            alpha_t = self.alpha(gamma_t, target_tensor=net_out)
            eps_t = net_out
            x_pred = 1. / alpha_t * (zt - sigma_t * eps_t)
        else:
            raise ValueError(self.parametrization)

        return x_pred

    def compute_error(self, net_out, gamma_t, eps):
        """Computes error, i.e. the most likely prediction of x."""
        eps_t = net_out
        if self.training and self.loss_type == 'l2':
            print("eps is ", np.shape(eps))
            print("eps_t is ", np.shape(eps_t))
            denom = (self.n_dims + self.in_node_nf) * eps_t.shape[1] #eps_t.shape[1] is batch size. denom is the number of all the features.
            error = sum_except_batch((eps - eps_t) ** 2) / denom
        else:
            error = sum_except_batch((eps - eps_t) ** 2)
        return error

    def log_constants_p_x_given_z0(self, x, node_mask):
        """Computes p(x|z0)."""
        batch_size = x.shape[0]

        n_nodes = node_mask.squeeze(2).sum(1)  # N has shape [B]
        assert n_nodes.shape == (batch_size,)
        degrees_of_freedom_x = (n_nodes - 1) * self.n_dims

        zeros = jnp.zeros((x.shape[0], 1))
        gamma_0 = self.gamma(zeros)

        # Recall that sigma_x = sqrt(sigma_0^2 / alpha_0^2) = SNR(-0.5 gamma_0).
        log_sigma_x = 0.5 * gamma_0.reshape(batch_size)

        return degrees_of_freedom_x * (- log_sigma_x - 0.5 * np.log(2 * np.pi))

    def sample_p_xh_given_z0(self, z0, node_mask, edge_mask, context, fix_noise=False):
        """Samples x ~ p(x|z0)."""
        zeros = jnp.zeros(size=(z0.shape[0], 1))
        gamma_0 = self.gamma(zeros)
        # Computes sqrt(sigma_0^2 / alpha_0^2)
        sigma_x = self.SNR(-0.5 * gamma_0).unsqueeze(1)
        net_out = self.phi(z0, zeros, node_mask, edge_mask, context)

        # Compute mu for p(zs | zt).
        mu_x = self.compute_x_pred(net_out, z0, gamma_0)
        xh = self.sample_normal(mu=mu_x, sigma=sigma_x, node_mask=node_mask, fix_noise=fix_noise)

        x = xh[:, :, :self.n_dims]

        h_int = z0[:, :, -1:] if self.include_charges else jnp.zeros(0)
        x, h_cat, h_int = self.unnormalize(x, z0[:, :, self.n_dims:-1], h_int, node_mask)

        h_cat = jax.nn.one_hot(jnp.argmax(h_cat, axis=2), self.num_classes) * node_mask
        h_int = jnp.round(h_int).astype(jnp.float32) * node_mask
        h = {'integer': h_int, 'categorical': h_cat}
        return x, h

    def sample_normal(self, mu, sigma, node_mask, fix_noise=False):
        """Samples from a Normal distribution."""
        bs = 1 if fix_noise else mu.shape[0]
        eps = self.sample_combined_position_feature_noise(bs, mu.shape[1], node_mask)
        return mu + sigma * eps

    def log_pxh_given_z0_without_constants(
            self, x, h, z_t, gamma_0, eps, net_out, node_mask, epsilon=1e-10):
        # Discrete properties are predicted directly from z_t.
        z_h_cat = z_t[:, :, self.n_dims:-1] if self.include_charges else z_t[:, :, self.n_dims:]
        z_h_int = z_t[:, :, -1:] if self.include_charges else jnp.zeros(0)

        # Take only part over x.
        eps_x = eps[:, :, :self.n_dims]
        net_x = net_out[:, :, :self.n_dims]

        # Compute sigma_0 and rescale to the integer scale of the data.
        sigma_0 = self.sigma(gamma_0, target_tensor=z_t)
        sigma_0_cat = sigma_0 * self.norm_values[1]
        sigma_0_int = sigma_0 * self.norm_values[2]

        # Computes the error for the distribution N(x | 1 / alpha_0 z_0 + sigma_0/alpha_0 eps_0, sigma_0 / alpha_0),
        # the weighting in the epsilon parametrization is exactly '1'.
        log_p_x_given_z_without_constants = -0.5 * self.compute_error(net_x, gamma_0, eps_x)

        # Compute delta indicator masks.
        h_integer = jnp.round(h['integer'] * self.norm_values[2] + self.norm_biases[2]).astype(jnp.float32)
        onehot = h['categorical'] * self.norm_values[1] + self.norm_biases[1]

        estimated_h_integer = z_h_int * self.norm_values[2] + self.norm_biases[2]
        estimated_h_cat = z_h_cat * self.norm_values[1] + self.norm_biases[1]
        assert h_integer.shape == estimated_h_integer.shape

        h_integer_centered = h_integer - estimated_h_integer

        # Compute integral from -0.5 to 0.5 of the normal distribution
        # N(mean=h_integer_centered, stdev=sigma_0_int)
        log_ph_integer = jnp.log(
            cdf_standard_gaussian((h_integer_centered + 0.5) / sigma_0_int)
            - cdf_standard_gaussian((h_integer_centered - 0.5) / sigma_0_int)
            + epsilon)
        log_ph_integer = sum_except_batch(log_ph_integer * node_mask)

        # Centered h_cat around 1, since onehot encoded.
        centered_h_cat = estimated_h_cat - 1

        # Compute integrals from 0.5 to 1.5 of the normal distribution
        # N(mean=z_h_cat, stdev=sigma_0_cat)
        log_ph_cat_proportional = jnp.log(
            cdf_standard_gaussian((centered_h_cat + 0.5) / sigma_0_cat)
            - cdf_standard_gaussian((centered_h_cat - 0.5) / sigma_0_cat)
            + epsilon)

        # Normalize the distribution over the categories.
        log_Z = jax.scipy.special.logsumexp(log_ph_cat_proportional, axis=2, keepdims=True)
        log_probabilities = log_ph_cat_proportional - log_Z

        # Select the log_prob of the current category usign the onehot
        # representation.
        log_ph_cat = sum_except_batch(log_probabilities * onehot * node_mask)

        # Combine categorical and integer log-probabilities.
        log_p_h_given_z = log_ph_integer + log_ph_cat

        # Combine log probabilities for x and h.
        log_p_xh_given_z = log_p_x_given_z_without_constants + log_p_h_given_z

        return log_p_xh_given_z

    def compute_loss(self, x, h, node_mask, edge_mask, context, t0_always):
        """Computes an estimator for the variational lower bound, or the simple loss (MSE)."""

        # This part is about whether to include loss term 0 always.
        if t0_always:
            # loss_term_0 will be computed separately.
            # estimator = loss_0 + loss_t,  where t ~ U({1, ..., T})
            lowest_t = 1
        else:
            # estimator = loss_t,           where t ~ U({0, ..., T})
            lowest_t = 0

        # Sample a timestep t.
        print("DL2: hack here, for the key")
        key=jax.random.PRNGKey(42)
        t_int = jax.random.randint(key,
            minval=lowest_t, maxval=self.T + 1, shape=(x.shape[0], 1)).astype(jnp.float32)
        s_int = t_int - 1
        t_is_zero = (t_int == 0).astype(jnp.float32)  # Important to compute log p(x | z0).

        # Normalize t to [0, 1]. Note that the negative
        # step of s will never be used, since then p(x | z0) is computed.
        s = s_int / self.T
        t = t_int / self.T

        # Compute gamma_s and gamma_t via the network.
        gamma_s = self.inflate_batch_array(self.gamma(s), x)
        gamma_t = self.inflate_batch_array(self.gamma(t), x)

        # Compute alpha_t and sigma_t from gamma.
        # print("gamma_t size is ", gamma_t.shape)
        # print("x size is",x.shape)
        alpha_t = self.alpha(gamma_t, x)
        sigma_t = self.sigma(gamma_t, x)

        # Sample zt ~ Normal(alpha_t x, sigma_t)
        eps = self.sample_combined_position_feature_noise(
            n_samples=x.shape[0], n_nodes=x.shape[1], node_mask=node_mask)

        # Concatenate x, h[integer] and h[categorical].



        
        xh = jnp.concatenate([a for a in [x, h['categorical'], h['integer']] if a.ndim>2], axis=2)
        # print("x shape is ",x.shape)
        # print("x categorical shape is ",h['categorical'].shape)
        # print("x integer shape is ",h['integer'].shape)
        # print("xh shape is ",xh.shape)
        # print("alpha_t shape is ",alpha_t.shape)
        # print("sigma_t shape is ",sigma_t.shape)
        # print("eps shape is ",eps.shape)

        # Sample z_t given x, h for timestep t, from q(z_t | x, h)
        z_t = alpha_t * xh + sigma_t * eps

        diffusion_utils.assert_mean_zero_with_mask(z_t[:, :, :self.n_dims], node_mask)

        # Neural net prediction.
        net_out = self.phi(z_t, t, node_mask, edge_mask, context)

        # Compute the error.
        error = self.compute_error(net_out, gamma_t, eps)

        if self.training and self.loss_type == 'l2':
            SNR_weight = jnp.ones_like(error)
        else:
            # Compute weighting with SNR: (SNR(s-t) - 1) for epsilon parametrization.
            SNR_weight = (self.SNR(gamma_s - gamma_t) - 1).squeeze(1).squeeze(1)
        assert error.shape == SNR_weight.shape
        loss_t_larger_than_zero = 0.5 * SNR_weight * error

        # The _constants_ depending on sigma_0 from the
        # cross entropy term E_q(z0 | x) [log p(x | z0)].
        neg_log_constants = -self.log_constants_p_x_given_z0(x, node_mask)

        # Reset constants during training with l2 loss.
        if self.training and self.loss_type == 'l2':
            neg_log_constants = jnp.zeros_like(neg_log_constants)

        # The KL between q(z1 | x) and p(z1) = Normal(0, 1). Should be close to zero.
        kl_prior = self.kl_prior(xh, node_mask)

        # Combining the terms
        if t0_always:
            loss_t = loss_t_larger_than_zero
            num_terms = self.T  # Since t=0 is not included here.
            estimator_loss_terms = num_terms * loss_t

            # Compute noise values for t = 0.
            t_zeros = jnp.zeros_like(s)
            gamma_0 = self.inflate_batch_array(self.gamma(t_zeros), x)
            alpha_0 = self.alpha(gamma_0, x)
            sigma_0 = self.sigma(gamma_0, x)

            # Sample z_0 given x, h for timestep t, from q(z_t | x, h)
            eps_0 = self.sample_combined_position_feature_noise(
                n_samples=x.shape[0], n_nodes=x.shape[1], node_mask=node_mask)
            z_0 = alpha_0 * xh + sigma_0 * eps_0

            net_out = self.phi(z_0, t_zeros, node_mask, edge_mask, context)

            loss_term_0 = -self.log_pxh_given_z0_without_constants(
                x, h, z_0, gamma_0, eps_0, net_out, node_mask)

            assert kl_prior.shape == estimator_loss_terms.shape
            assert kl_prior.shape == neg_log_constants.shape
            assert kl_prior.shape == loss_term_0.shape

            loss = kl_prior + estimator_loss_terms + neg_log_constants + loss_term_0

        else:
            # Computes the L_0 term (even if gamma_t is not actually gamma_0)
            # and this will later be selected via masking.
            loss_term_0 = -self.log_pxh_given_z0_without_constants(
                x, h, z_t, gamma_t, eps, net_out, node_mask)

            t_is_not_zero = 1 - t_is_zero

            loss_t = loss_term_0 * t_is_zero.squeeze() + t_is_not_zero.squeeze() * loss_t_larger_than_zero

            # Only upweigh estimator if using the vlb objective.
            if self.training and self.loss_type == 'l2':
                estimator_loss_terms = loss_t
            else:
                num_terms = self.T + 1  # Includes t = 0.
                estimator_loss_terms = num_terms * loss_t

            assert kl_prior.shape == estimator_loss_terms.shape
            assert kl_prior.shape == neg_log_constants.shape

            loss = kl_prior + estimator_loss_terms + neg_log_constants

        assert len(loss.shape) == 1, f'{loss.shape} has more than only batch dim.'

        return loss, {'t': t_int.squeeze(), 'loss_t': loss.squeeze(),
                      'error': error.squeeze()}



    def sample_p_zs_given_zt(self, s, t, zt, node_mask, edge_mask, context, fix_noise=False):
        """Samples from zs ~ p(zs | zt). Only used during sampling."""
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = \
            self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zt)

        sigma_s = self.sigma(gamma_s, target_tensor=zt)
        sigma_t = self.sigma(gamma_t, target_tensor=zt)

        # Neural net prediction.
        eps_t = self.phi(zt, t, node_mask, edge_mask, context)

        # Compute mu for p(zs | zt).
        diffusion_utils.assert_mean_zero_with_mask(zt[:, :, :self.n_dims], node_mask)
        diffusion_utils.assert_mean_zero_with_mask(eps_t[:, :, :self.n_dims], node_mask)
        mu = zt / alpha_t_given_s - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_t

        # Compute sigma for p(zs | zt).
        sigma = sigma_t_given_s * sigma_s / sigma_t

        # Sample zs given the paramters derived from zt.
        zs = self.sample_normal(mu, sigma, node_mask, fix_noise)

        # Project down to avoid numerical runaway of the center of gravity.
        zs = jnp.concatenate(
            [diffusion_utils.remove_mean_with_mask(zs[:, :, :self.n_dims],
                                                   node_mask),
             zs[:, :, self.n_dims:]], axis=2
        )
        return zs

    def sample_combined_position_feature_noise(self, n_samples, n_nodes, node_mask):
        """
        Samples mean-centered normal noise for z_x, and standard normal noise for z_h.
        """
        z_x = utils.sample_center_gravity_zero_gaussian_with_mask(
            size=(n_samples, n_nodes, self.n_dims), device=node_mask.device,
            node_mask=node_mask)
        z_h = utils.sample_gaussian_with_mask(
            size=(n_samples, n_nodes, self.in_node_nf), device=node_mask.device,
            node_mask=node_mask)
        # print("z_h shape is",z_h.shape)
        z = jnp.concatenate([z_x, z_h], axis=2)
        return z

    # 
    def sample(self, n_samples, n_nodes, node_mask, edge_mask, context, fix_noise=False):
        """
        Draw samples from the generative model.
        """
        if fix_noise:
            # Noise is broadcasted over the batch axis, useful for visualizations.
            z = self.sample_combined_position_feature_noise(1, n_nodes, node_mask)
        else:
            z = self.sample_combined_position_feature_noise(n_samples, n_nodes, node_mask)

        diffusion_utils.assert_mean_zero_with_mask(z[:, :, :self.n_dims], node_mask)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s in reversed(range(0, self.T)):
            s_array = jnp.full((n_samples, 1), fill_value=s)
            t_array = s_array + 1
            s_array = s_array / self.T
            t_array = t_array / self.T

            z = self.sample_p_zs_given_zt(s_array, t_array, z, node_mask, edge_mask, context, fix_noise=fix_noise)

        # Finally sample p(x, h | z_0).
        x, h = self.sample_p_xh_given_z0(z, node_mask, edge_mask, context, fix_noise=fix_noise)

        diffusion_utils.assert_mean_zero_with_mask(x, node_mask)

        max_cog = jnp.sum(x, axis=1, keepaxis=True).abs().max().item()
        if max_cog > 5e-2:
            print(f'Warning cog drift with error {max_cog:.3f}. Projecting '
                  f'the positions down.')
            x = diffusion_utils.remove_mean_with_mask(x, node_mask)

        return x, h

    
    def sample_chain(self, n_samples, n_nodes, node_mask, edge_mask, context, keep_frames=None):
        """
        Draw samples from the generative model, keep the intermediate states for visualization purposes.
        """
        z = self.sample_combined_position_feature_noise(n_samples, n_nodes, node_mask)

        diffusion_utils.assert_mean_zero_with_mask(z[:, :, :self.n_dims], node_mask)

        if keep_frames is None:
            keep_frames = self.T
        else:
            assert keep_frames <= self.T
        chain = jnp.zeros((keep_frames,) + z.shape)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s in reversed(range(0, self.T)):
            s_array = jnp.full((n_samples, 1), fill_value=s, device=z.device)
            t_array = s_array + 1
            s_array = s_array / self.T
            t_array = t_array / self.T

            z = self.sample_p_zs_given_zt(
                s_array, t_array, z, node_mask, edge_mask, context)

            diffusion_utils.assert_mean_zero_with_mask(z[:, :, :self.n_dims], node_mask)

            # Write to chain tensor.
            write_index = (s * keep_frames) // self.T
            chain[write_index] = self.unnormalize_z(z, node_mask)

        # Finally sample p(x, h | z_0).
        x, h = self.sample_p_xh_given_z0(z, node_mask, edge_mask, context)

        diffusion_utils.assert_mean_zero_with_mask(x[:, :, :self.n_dims], node_mask)

        xh = jnp.concatenate([x, h['categorical'], h['integer']], axis=2)
        chain[0] = xh  # Overwrite last frame with the resulting x and h.

        chain_flat = chain.reshape(n_samples * keep_frames, *z.shape[1:])

        return chain_flat

    def log_info(self):
        """
        Some info logging of the model.
        """
        gamma_0 = self.gamma(jnp.zeros(1))
        gamma_1 = self.gamma(jnp.ones(1))

        log_SNR_max = -gamma_0
        log_SNR_min = -gamma_1

        info = {
            'log_SNR_max': log_SNR_max.item(),
            'log_SNR_min': log_SNR_min.item()}
        print(info)

        return info
