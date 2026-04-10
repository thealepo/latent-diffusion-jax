'''
encoders.py

Builds the wrappers for the pre-trained CLIP Text Encoder and the VAE models from Huggingface.
Both models are loaded in JAX/Flax.
'''

import jax
import jax.numpy as jnp
from transformers import CLIPTokenizer
from transformers.model.clip.modeling_flax_clip import FlaxCLIPTextModel
from diffusers import FlaxAutoencoderKL
from typing import Tuple

# Loading the models
def load_clip(model_id: str = 'openai/clip-vit-large-patch14') -> Tuple[CLIPTokenizer , FlaxCLIPTextModel , dict]:
    ''' Load a CLIP text encoder from Huggingface. '''
    tokenizer = CLIPTokenizer.from_pretrained(model_id)
    text_encoder = FlaxCLIPTextModel.from_pretrained(model_id , dtype=jnp.float32)
    return tokenizer , text_encoder , text_encoder.params

def load_vae(model_id: str = 'runwayml/stable-diffusion-v1-5' , subfolder: str = 'vae') -> Tuple[FlaxAutoencoderKL , dict]:
    ''' Load a VAE model from Huggingface. '''
    vae , vae_params = FlaxAutoencoderKL.from_pretrained(
        model_id , subfolder=subfolder , from_pt=True , dtype=jnp.float32
    )
    return vae , vae_params

# Helpers
def get_text_embeddings(prompts: List[str] , tokenizer: CLIPTokenizer , text_encoder: FlaxCLIPTextModel , text_encoder_params: dict) -> jnp.ndarray:
    ''' 
    Tokenize and encode a list of prompts into text embeddings.
    Returns a numpy array of shape (len(prompts) , seq_len , embed_dim)
    '''
    inputs = tokenizer(prompts , padding='max_length' , max_length=tokenizer.model_max_length , truncation=True , return_tensors='np')
    return text_encoder(inputs.input_ids , params=text_encoder_params)[0]

@jax.jit
def encode_to_latents(pixel_values: jnp.ndarray , vae: FlaxAutoencoderKL , vae_params: dict , rng: jnp.ndarray) -> jnp.ndarray:
    ''' 
    Encode a batch of pixel values into a VAE latent space. 
    Returns a numpy array of shape (B , H/8 , W/8 , 4)
    '''
    latent_dist = vae.apply({'params': vae_params} , pixel_values , method=vae.encode).latent_dist
    latents = latent_dist.sample(rng)
    return latents * vae.config.scaling_factor

@jax.jit
def decode_latents(latents: jnp.ndarray , vae: FlaxAutoencoderKL , vae_params: dict) -> jnp.ndarray:
    ''' 
    Decode a batch of latents into a pixel space.
    Returns a numpy array of shape (B , H , W , C)
    '''
    unscaled = latents / vae.config.scaling_factor
    decoded = vae.apply({'params': vae_params} , unscaled , method=vae.decode).sample
    decoded = jnp.clip(decoded , -1.0 , 1.0)
    return (decoded + 1.0) / 2.0