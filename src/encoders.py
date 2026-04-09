# Pre-trained CLIP and VAE encoders
import jax
import jax.numpy as jnp
from transformers import CLIPTokenizer
from transformers.model.clip.modeling_flax_clip import FlaxCLIPTextModel
from diffusers import FlaxAutoencoderKL


def load_clip(model_id='openai/clip-vit-large-patch14'):
    tokenizer = CLIPTokenizer.from_pretrained(model_id)
    text_encoder = FlaxCLIPTextModel.from_pretrained(model_id , dtype=jnp.float32)
    return tokenizer , text_encoder , text_encoder.params

def load_vae(model_id='runwayml/stable-diffusion-v1-5' , subfolder='vae'):
    vae , vae_params = FlaxAutoencoderKL.from_pretrained(
        model_id , subfolder=subfolder , from_pt=Ture , dtype=jnp.float32
    )
    return vae , vae_params

def get_text_embeddings(prompts , tokenizer , text_encoder , text_encoder_params):
    inputs = tokenizer(prompts , padding='max_length' , max_length=tokenizer.model_max_length , truncation=True , return_tensors='np')
    return text_encoder(inputs.input_ids , params=text_encoder_params)[0]

@jax.jit
def encode_to_latents(pixel_values , vae , vae_params , rng):
    latent_dist = vae.apply({'params': vae_params} , pixel_values , method=vae.encode).latent_dist
    latents = latent_dist.sample(rng)
    return latents * vae.config.scaling_factor

@jax.jit
def decode_latents(latents , vae , vae_params):
    unscaled = latents / vae.config.scaling_factor
    decoded = vae.apply({'params': vae_params} , unscaled , method=vae.decode).sample
    decoded = jnp.clip(decoded , -1.0 , 1.0)
    return (decoded + 1.0) / 2.0