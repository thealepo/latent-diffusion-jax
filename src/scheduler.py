'''
scheduler.py

DDPM (Denoising Diffusion Probabilistic Models) scheduler.

Implements linear beta schedule.

Ho et. al., "Denoising Diffusion Probabilistic Models"
    https://arxiv.org/abs/2006.11239
'''

import jax.numpy as jnp
from einops import rearrange

class DDPMScheduler:
    '''
    Linear-scheduled DDPM noise scheduler.
    '''
    def __init__(self , num_timesteps: int = 1000 , beta_start: float = 0.00085 , beta_end: float = 0.012) -> None:
        self.num_timesteps = num_timesteps

        # β_t: linear schedule from beta_start to beta_end
        self.betas = jnp.linspace(beta_start , beta_end , num_timesteps)

        # α_t = 1 - β_t
        self.alphas = 1.0 - self.betas
        # ᾱ_t = ∏_{s=1}^{t} α_s  (cumulative product of alphas)
        self.alpha_bar = jnp.cumprod(self.alphas , axis=0)

        # Precomputed terms
        self.alpha_bar_prev = jnp.concatenate([jnp.array([1.0]) , self.alpha_bar[:-1]])
        self.sqrt_alpha_bar = jnp.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = jnp.sqrt(1.0 - self.alpha_bar)
        self.sqrt_recip_alpha = jnp.sqrt(1.0 / self.alphas)
        self.posterior_variance = (
            self.betas * (1.0 - self.alpha_bar_prev) / (1.0 - self.alpha_bar)
        )

    # Forward Process
    def add_noise(self , x_0: jnp.ndarray , noise: jnp.ndarray , t: jnp.ndarray) -> jnp.ndarray:
        '''
        Sample x_t from the forward diffusion process q(x_t | x_0)
        Uses the closed-form solution:
            x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε,  ε ~ N(0, I)
        '''
        sqrt_alpha_bar = self.sqrt_alpha_bar[t]
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar[t]

        # Reshape for broadcasting
        sqrt_alpha_bar = rearrange(sqrt_alpha_bar , 'b -> b 1 1 1')
        sqrt_one_minus_alpha_bar = rearrange(sqrt_one_minus_alpha_bar , 'b -> b 1 1 1')

        return sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise

    # Reverse Process
    def step(self , noise_pred: jnp.ndarray , x_t: jnp.ndarray , t: int) -> Tuple[jnp.ndarray , jnp.ndarray | float]:
        '''
        Computes the mean and variance of the reverse diffusion process p(x_{t-1} | x_t)
        Implements the DDPM reverse step:
            μ_θ(x_t, t) = (1/√α_t) * (x_t - β_t / √(1 - ᾱ_t) * ε_θ(x_t, t))
        '''
        alpha = self.alphas[t]
        alpha_bar = self.alpha_bar[t]
        beta = self.betas[t]

        coeff = beta / jnp.sqrt(1.0 - alpha_bar)
        mean = (1.0 / jnp.sqrt(alpha)) * (x_t - coeff * noise_pred)

        if t > 0:
            variance = self.posterior_variance[t]
            return mean , variance
        else:
            return mean , 0.0