# **Mini-Latent Diffusion Implementation in JAX**

A JAX implementation of **Latent Diffusion**. Visit the ***[Google Colab Notebook](https://colab.research.google.com/drive/1YZqdrmZ4bmJNQZvfoc4FirldroHJgnY2?usp=sharing).***

![LDM Diagram](./assets/LDM_Dia.png)

---

## Overview

This repository implements the core mechanisms of diffusion, including the **Scheduler** and the **U-Net Architecture**. The supporting **Variational Autoencoder (VAE)** and **CLIP** model are frozen and imported. Therefore, you can consider the core paper this code follows as *[Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239)*, with some supporting ideas from *[High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/pdf/2112.10752)*.

Another aspect of this project is the implementation being on **JAX**, and the **Flax NNX** API.

---

## Some Generated Images!

While a full training cycle was not done, we still have some previews on the capabilities of the model, having trained on ***120k MS-COCO*** images for 30 epochs. Enjoy the 3 test samples, which include the prompts `a cat on the grass`, `a red car on a highway`, and `a sunset over the city`, respectively.