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

def train(
    unet,
    optimizer,
    scheduler,
    dataloader,
    encode_fn,
    text_embed_fn,
    inference_fn,
    checkpoint_dir,
    num_epochs=50,
    cfg_dropout=0.1,
    log_every=50,
    save_every=5,
    sample_every=10,
    start_epoch=0,
):
wandb.init(project='mini-latent-diffusion-jax')

rng = jax.random.PRNGKey(42)
global_step = start_epoch * len(dataloader)

sample_prompts = [
    'a cat in the grass',
    'a red car on the highway',
    'a sunset over the city'
]

for epoch in tqdm(range(start_epoch , num_epochs) , desc='Epochs'):
    epoch_losses = []

    for _ , (images,captions) in tqdm(enumerate(dataloader) , total=len(dataloader) , desc=f'Epoch {epoch+1}' , leave=False):
        k1 , k2 , k3 = jax.random.split(rng , 3)

        latents = encode_fn(jnp.array(images) , k2)
        clip_embeddings = jnp.array(text_embed_fn(captions))

        loss = train_step(
            unet,
            optimizer,
            scheduler,
            latents,
            clip_embeddings,
            cfg_dropout,
            rngs=k3
        )
        loss_value = float(loss)
        epoch_losses.append(loss_value)

        if global_step % log_every == 0:
            wandb.log({'train/loss': loss_value , 'global_step':global_step})
        global_step += 1

    avg_loss = np.mean(epoch_losses)
    wandb.log({"train/epoch_loss": avg_loss , "epoch": epoch+1})
    print(f'Epoch {epoch+1} | avg_loss = {avg_loss:.4f}')

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

    if (epoch+1) % save_every == 0:
        _save_checkpoint(unet , optimizer , epoch , checkpoint_dir)

wandb.finish()


# Checkpointing helper functions
def _save_checkpoint(unet , optimizer , epoch , checkpoint_dir):
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
def load_checkpoint(epoch , checkpoint_dir , unet , optimizer):
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