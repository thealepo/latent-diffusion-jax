import jax.numpy as jnp
from einops import rearrange

class DDPMScheduler:
    def __init__(self , num_timesteps=1000 , beta_start=0.00085 , beta_end=0.012):
        self.num_timesteps = num_timesteps

        # Linear schedule for betas
        self.betas = jnp.linspace(beta_start , beta_end , num_timesteps)

        # Alphas
        self.alphas = 1.0 - self.betas
        self.alpha_bar = jnp.cumprod(self.alphas , axis=0)

        # Precomputed terms
        self.alpha_bar_prev = jnp.concatenate([jnp.array([1.0]) , self.alpha_bar[:-1]])
        self.sqrt_alpha_bar = jnp.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = jnp.sqrt(1.0 - self.alpha_bar)
        self.sqrt_recip_alpha = jnp.sqrt(1.0 / self.alphas)
        self.posterior_variance = (
            self.betas * (1.0 - self.alpha_bar_prev) / (1.0 - self.alpha_bar)
        )

    def add_noise(self , x_0 , noise , t):
        sqrt_alpha_bar = self.sqrt_alpha_bar[t]
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar[t]

        sqrt_alpha_bar = rearrange(sqrt_alpha_bar , 'b -> b 1 1 1')
        sqrt_one_minus_alpha_bar = rearrange(sqrt_one_minus_alpha_bar , 'b -> b 1 1 1')

        return sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise

    def step(self , noise_pred , x_t , t):
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