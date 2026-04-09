from typing import List

import jax
import jax.numpy as jnp
import numpy as numpy

from src.unet import UNet
from src.scheduler import DDPMScheduler

def inference(
    unet,
    prompts,
    scheduler,
    num_steps,
    text_embed_fn,
    decode_fn,
    cfg_scale=7.5,
    rngs=jax.random.PRNGKey(42),
    return_trajectory=False,
    latent_size=32,
    latent_channels=4
):
B = len(prompts)

# Conditional and unconditional text embeddings
conditional_embeddings = jnp.array(text_embed_fn(prompts))
unconditional_embeddings = jnp.array(text_embed_fn(['']*B))

rngs , noise_rng = jax.random.split(rngs , 2)
x_t = jax.random.normal(noise_rng , (B,latent_size,latent_size,latent_channels))

trajectory = []
for t_idx in reversed(range(num_steps)):
    t_batch = jnp.full((B,) , t_idx , dtype=jnp.int32)

    conditional_noise_pred = unet(x_t , t_batch , conditional_embeddings)
    unconditional_noise_pred = unet(x_t , t_batch , unconditional_embeddings)

    # CFG
    noise_pred = unconditional_noise_pred + cfg_scale * (conditional_noise_pred - unconditional_noise_pred)

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