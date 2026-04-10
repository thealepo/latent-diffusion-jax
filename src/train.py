'''
train.py

Main training loop and checkpointing for the Mini-Latent Diffusion model.

The main objective is to predict the noise at each timestep t:
    L = E_{x_0, ε, t} [ || ε - ε_θ(x_t, t, c) ||² ]
'''

import os
from typing import Callable, List, Tuple

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
def train_step(unet: UNet , optimizer: nnx.Optimizer , scheduler: DDPMScheduler , latents: jnp.ndarray , clip_embeddings: jnp.ndarray , cfg_dropout: float , rng: jax.Array) -> jnp.ndarray:
    '''
    Executes a single gradient update step.

    Randomly samples the timestep and noise, applies forward diffusion, and updates the UNet weights.
    '''
    B = latents.shape[0]
    k1 , k2 , k3 = jax.random.split(rng , 3)

    # Sampling random noise and timesteps
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

# Full Training
def train(
    unet: UNet,
    optimizer: nnx.Optimizer,
    scheduler: DDPMScheduler,
    dataloader: object,
    encode_fn: Callable[[jnp.ndarray , jax.Array] , jnp.ndarray],
    text_embed_fn: Callable[[List[str]] , np.ndarray],
    inference_fn: Callable,
    checkpoint_dir: str,
    num_epochs: int = 50,
    cfg_dropout: float = 0.1,
    log_every: int = 50,
    save_every: int = 5,
    sample_every: int = 10,
    start_epoch: int = 0,
) -> None:
    '''
    Full training loop for the latent diffusion UNet.

    In each batch, the images are encoded into the latent space with the VAE, text prompts are embedded with CLIP, and then a single gradient step is taken.
    Metrics are logged to WandB and every once in a while, an image is sampled and logged to WandB.
    '''
    wandb.init(project='mini-latent-diffusion-jax')

    rng = jax.random.PRNGKey(42)
    global_step = start_epoch * len(dataloader)

    # Prompts used for visualizing prohress across epochs
    sample_prompts = [
        'a cat in the grass',
        'a red car on the highway',
        'a sunset over the city'
    ]

    for epoch in tqdm(range(start_epoch , num_epochs) , desc='Epochs'):
        epoch_losses = []

        for _ , (images,captions) in tqdm(enumerate(dataloader) , total=len(dataloader) , desc=f'Epoch {epoch+1}' , leave=False):
            # Splitting rng into three distinct keys
            k1 , k2 , k3 = jax.random.split(rng , 3)

            # Encoding images into latents, strings into embeddings
            latents = encode_fn(jnp.array(images) , k2)
            clip_embeddings = jnp.array(text_embed_fn(captions))

            loss = train_step(
                unet,
                optimizer,
                scheduler,
                latents,
                clip_embeddings,
                cfg_dropout,
                rng=k3
            )
            loss_value = float(loss)
            epoch_losses.append(loss_value)

            if global_step % log_every == 0:
                wandb.log({'train/loss': loss_value , 'global_step':global_step})
            global_step += 1

        # Logging the per-epoch average loss
        avg_loss = np.mean(epoch_losses)
        wandb.log({"train/epoch_loss": avg_loss , "epoch": epoch+1})
        print(f'Epoch {epoch+1} | avg_loss = {avg_loss:.4f}')

        # Periodically generate and log the sample images
        if (epoch+1) % sample_every == 0:
            k4 , k5 = jax.random.split(rng , 2)
            imgs = inference_fn(
                unet,
                sample_prompts,
                scheduler,
                num_steps=1000,
                cfg_scale=7.5,
                rngs=k4,
                return_trajectory=False
            )
            wandb.log({"samples": [wandb.Image(img,caption=p) for img,p in zip(imgs,sample_prompts)] , "epoch": epoch + 1})

        # Periodically save the checkpoint
        if (epoch+1) % save_every == 0:
            _save_checkpoint(unet , optimizer , epoch , checkpoint_dir)

    wandb.finish()


# Checkpointing helper functions
def _save_checkpoint(unet: UNet , optimizer: nnx.Optimizer , epoch: int , checkpoint_dir: str) -> None:
    '''
    Saves the model and optimizer state to a checkpoint file.
    '''
    os.makedirs(checkpoint_dir , exist_ok=True)
    path = os.path.abspath(os.path.join(checkpoint_dir , f'epoch_{epoch:04d}'))
    ocp.StandardCheckpointer().save(
        path,
        {
            'unet': nnx.state(unet),
            'optimizer': nnx.state(optimizer),
            'epoch': epoch
        },
        force=True
    )
def load_checkpoint(epoch: int , checkpoint_dir: str , unet: UNet , optimizer: nnx.Optimizer) -> Tuple[UNet , nnx.Optimizer , int]:
    '''
    Loads the model and optimizer state from a checkpoint file.
    '''
    path = os.path.abspath(os.path.join(checkpoint_dir , f'epoch_{epoch:04d}'))
    target = {
        'unet': nnx.state(unet),
        'optimizer': nnx.state(optimizer),
        'epoch': 0
    }
    state = ocp.StandardCheckpointer().restore(path,target)

    nnx.update(unet , state['unet'])
    nnx.update(optimizer , state['optimizer'])
    return unet , optimizer , state['epoch']