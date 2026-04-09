import os

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import numpy as np
import orbax.checkpoint as ocp
import wandb
from tqdm.auto import tqdm

from src.unet import UNet
from src.scheduler import DDPMScheduler


# Single training step
@nnx.jit
def train_step(unet , optimizer , scheduler , latents , clip_embeddings , cfg_dropout , rng):
    B = latents.shape[0]
    k1 , k2 , k3 = jax.random.split(rng , 3)

    noise = jax.random.normal(k1 , latents.shape)
    timesteps = jax.random.randint(k2 , (B,) , 0 , scheduler.num_timesteps)
    x_t = scheduler.add_noise(latents , noise , timesteps)

    # CFG Dropout
    dropout_mask = jax.random.uniform(k3 , (B,1,1,1)) < cfg_dropout
    cfg_embeddings = jnp.where(dropout_mask , jnp.zeros_like(clip_embeddings) , clip_embeddings)

    def loss_fn(unet):
        noise_pred = unet(x_t , timesteps , cfg_embeddings)
        return jnp.mean((noise_pred - noise) ** 2)

    loss , grads = jax.value_and_grad(loss_fn)(unet)
    optimizer.update(unet , grads)
    return loss

