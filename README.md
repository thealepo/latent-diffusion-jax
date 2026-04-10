# **Mini-Latent Diffusion Implementation in JAX**

A JAX implementation of **Latent Diffusion**. Visit the ***[Google Colab Notebook](https://colab.research.google.com/drive/1YZqdrmZ4bmJNQZvfoc4FirldroHJgnY2?usp=sharing).***

![LDM Diagram](./assets/LDM_Dia.png)

---

## Overview

This repository implements the core mechanisms of diffusion, including the **Scheduler** and the **U-Net Architecture**. The supporting **Variational Autoencoder (VAE)** and **CLIP** model are frozen and imported. Therefore, you can consider the core paper this code follows as *[Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239)*, with some supporting ideas from *[High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/pdf/2112.10752)*.

The full pipeline includes noise scheduling, the main U-Net architecture (with cross-attention injected at lower levels), and classifier-free guidance, all written in **JAX/Flax NNX**.

---

## Architecture

This model was built from scratch in JAX/Flax NNX and follows the standard latent diffusion design:

- **DDPM Scheduler:** Linear beta schedule with the forward/reverse process mathematical constants. Implements `add_noise` (forward) and `step` (reverse) for training and inference respectively.
- **UNet:** The core model. Consists of symmetric encoder/decoder paths (with skip connections), a deep bottleneck, and a final projection.
    - **DoubleConv Blocks:** Two Conv -> GroupNorm -> SiLU layers.
    - **CrossAttention Blocks:** Multihead attention where the image features (query) attend to CLIP text embeddings (key/value). Applied at the deepest layers of the UNet.
    - **DownBlocks / UpBlocks:** DoubleConv + optional CrossAttention, with MaxPool downsampling and ConvTranspose upsampling.
    - **Timestep Embedding:** Encoding of integer timestep, passed through an MLP.
- **Frozen CLIP Encoder:** Encodes text prompts into 768-dim embeddings that condition the UNet via cross-attention. (`openai/clip-vit-large-patch14`)
- **Frozen VAE:** Compresses 256×256 images into 32×32×4 latents for training, and decodes generated latents back to pixel space. (`runwayml/stable-diffusion-v1-5`)
- **Classifier-Free Guidance (CFG):** Done during training so that model learns both conditional and unconditional denoising. At inference, both passes are compared: `ε = ε_uncond + scale * (ε_cond − ε_uncond)`.

---

## Some Generated Images!

While a full training cycle was not done, we still have some previews on the capabilities of the model, having trained on ***120k MS-COCO*** images for 30 epochs. Enjoy the 3 test samples, which include the prompts `a sunset over the city`, `a red car on a highway`, and `a cat in the grass` respectively. The images are shown from epochs 10, 20, and 30.

### "A Sunset over the City"

This was my favorite! The 30th epoch representation for this prompt shows this artistic image. Although details largely lack to consider it realistic (off by a lot), you can definitely consider it beautiful "art".

![CITY](./assets/A_SUNSET_OVER_THE_CITY_30.png)

### "A Red Car on a Highway"

We can see the model follow the prompts we gave it. As the epochs pass, you can see the "essence" of a car start being actualized, with some car details starting to show up. However, it still remained relatively abstract.

![RED](./assets/A_RED_CAR_ON_A_HIGHWAY.png)

### "A Cat in the Grass"

This would be what you can consider out "worst" results. Due to the cat's fine features, the representation remained the most abstract, though it is interesting to note the level of detail the grass started getting.

![CAT](./assets/A_CAT_IN_THE_GRASS.png)

---

## Notes

These results serve as a proof-of-concept, not a production model. The goal of this project was to build a latent diffusion system from the ground up in JAX. 30 epochs on a limited compute budget is enough to see the model develop real semantic structure, which is the point.

For a deeper walkthrough of the theory and implementation, check out the accompanying lectures:

- [Lecture (Theory)](https://youtu.be/qHOgvKH1Gi0?si=3FkU_mwkRP2j1V_Y)
- [Implementation (Code)](https://youtu.be/xPImI7d5IvY?si=uLmnFJO0G5o-j81a)
- [Google Colab Notebook](https://colab.research.google.com/drive/1YZqdrmZ4bmJNQZvfoc4FirldroHJgnY2?usp=sharing)
