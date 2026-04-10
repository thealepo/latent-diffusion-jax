'''
inference.py

DDPM reverse process with classifier-free guidance (CFG).

Iterates from t=T-1 to t=0, predicting and removing noise at each step.
'''

from typing import List

import jax
import jax.numpy as jnp
import numpy as np

from src.unet import UNet
from src.scheduler import DDPMScheduler

def inference(
    unet: UNet,
    prompts: List[str],
    scheduler: DDPMScheduler,
    num_steps: int,
    text_embed_fn: Callable[[List[str]] , np.ndarray],
    decode_fn: Callable[[jnp.ndarray] , jnp.ndarray],
    cfg_scale: float =7.5,
    rngs: jax.Array = jax.random.PRNGKey(42),
    return_trajectory: bool = False,
    latent_size: int = 32,
    latent_channels: int = 4
) -> Union[List[np.ndarray] , Tuple[List[np.ndarray] , List[Tuple[int , np.ndarray]]]]:
    '''
    DDPM reverse process with classifier-free guidance (CFG).
    '''
    B = len(prompts)

    # Text embeddings for conditional (prompt) and unconditional (null) prompts
    conditional_embeddings = jnp.array(text_embed_fn(prompts))
    unconditional_embeddings = jnp.array(text_embed_fn(['']*B))

    # Sample the initial Gaussian noise x_T ~ N(0,I)
    rngs , noise_rng = jax.random.split(rngs , 2)
    x_t = jax.random.normal(noise_rng , (B,latent_size,latent_size,latent_channels))

    # Reverse diffusion loop: t = T-1 -> 0
    trajectory = []
    for t_idx in reversed(range(num_steps)):
        t_batch = jnp.full((B,) , t_idx , dtype=jnp.int32)

        # Predict the noise for both conditional and unconditional paths
        conditional_noise_pred = unet(x_t , t_batch , conditional_embeddings)
        unconditional_noise_pred = unet(x_t , t_batch , unconditional_embeddings)

        # Classifier-free guidance (CFG)
        noise_pred = unconditional_noise_pred + cfg_scale * (conditional_noise_pred - unconditional_noise_pred)

        # Denoised mean
        mean , variance = scheduler.step(noise_pred , x_t , t_idx)

        if t_idx > 0:
            rngs , step_rng = jax.random.split(rngs)
            noise = jax.random.normal(step_rng , x_t.shape)
            x_t = mean + jnp.sqrt(variance) * noise
        else:
            x_t = mean

        if return_trajectory and t_idx % 100 == 0:
            trajectory.append((t_idx , np.array(x_t)))

    # Decoding latents
    pixel_images = np.array(decode_fn(x_t))
    result = []
    for i in range(B):
        img = (pixel_images[i]*255.0).clip(0,255).astype(np.uint8)
        if img.ndim == 3 and img.shape[0] == 3:
            img = np.transpose(img , (1,2,0))
        result.append(img)

    return (result , trajectory) if return_trajectory else result