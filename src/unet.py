from typing import List

import jax.numpy as jnp
import flax.nnx as nnx
from einops import rearrange


# Timestep Embedding
def get_timestep_embedding(timesteps , embedding_dim: int = 256):
    half_dim = embedding_dim // 2
    exponent = -jnp.log(10000.0) * jnp.arange(half_dim) / half_dim
    emb = jnp.exp(exponent)
    emb = timesteps[: , None] * emb[None , :]
    return jnp.concatenate([jnp.sin(emb) , jnp.cos(emb)] , axis=-1)

# Double Convolution Block
class DoubleConv(nnx.Module):
    def __init__(self , in_channels , out_channels , time_embed_dim , * , rngs):
        self.conv1 = nnx.Conv(in_channels , out_channels , kernel_size=(3,3) , padding='SAME' , rngs=rngs)
        self.norm1 = nnx.GroupNorm(num_groups=32 , num_features=out_channels , rngs=rngs)
        self.conv2 = nnx.Conv(out_channels , out_channels , kernel_size=(3,3) , padding='SAME' , rngs=rngs)
        self.norm2 = nnx.GroupNorm(num_groups=32 , num_features=out_channels , rngs=rngs)

        self.time_mlp = nnx.Linear(time_embed_dim , out_channels , rngs=rngs)

    def __call__(self , x , time_embed):
        # First Convolution
        x = self.conv1(x)
        x = self.norm1(x)
        x = nnx.silu(x)

        # Time embeddings
        t = self.time_mlp(nnx.silu(time_embed))
        t = rearrange(t , 'b c -> b 1 1 c')
        x = x + t

        # Second Convolution
        x = self.conv2(x)
        x = self.norm2(x)
        x = nnx.silu(x)

        return x

# Cross Attention Block
# Query (Image) attends to Key (Text) and Value (Text)
class CrossAttention(nnx.Module):
    def __init__(self , channels , context_dim , n_heads: int = 8 , * , rngs):
        self.n_heads = n_heads
        self.head_dim = channels // n_heads
        self.norm = nnx.GroupNorm(num_groups=32 , num_features=channels , rngs=rngs)

        # Attention Matrices
        self.wq = nnx.Linear(channels , channels , rngs=rngs)
        self.wk = nnx.Linear(context_dim , channels , rngs=rngs)
        self.wv = nnx.Linear(context_dim , channels , rngs=rngs)
        self.wo = nnx.Linear(channels , channels , rngs=rngs)

    def __call__(self , x , context):
        B , H , W , C = x.shape
        residual = x

        h = self.norm(x)
        h_flat = rearrange(h , 'b h w c -> b (h w) c')

        Q = self.wq(h_flat)
        K = self.wk(context)
        V = self.wv(context)

        def mha_reshape(tensor):
            return rearrange(tensor , 'b n (head d) -> b head n d' , head=self.n_heads)
        Q , K , V = map(mha_reshape , (Q , K , V))

        scale = 1.0 / jnp.sqrt(self.head_dim)
        attention = jnp.einsum('b h i d , b h j d -> b h i j' , Q , K) * scale
        attention = nnx.softmax(attention , axis=-1)
        out = jnp.einsum('b h i j , b h j d -> b h i d' , attention , V)

        out = rearrange(out , 'b h n d -> b n (h d)')
        out = self.wo(out)
        out = out.reshape((B , H , W , C))

        return out + residual

# Downsample Block
# Consists of DoubleConv, CrossAttention, followed by a max pool
class DownBlock(nnx.Module):
    def __init__(self , in_channels , out_channels , time_embed_dim , context_dim , use_attn , * , rngs):
        self.double_conv = DoubleConv(in_channels , out_channels , time_embed_dim , rngs=rngs)
        self.attention = CrossAttention(out_channels , context_dim , rngs=rngs) if use_attn else None

    def __call__(self , x , t , context):
        x = self.double_conv(x , t)
        if self.attention:
            x = self.attention(x , context)
        
        skip = x
        x = nnx.max_pool(x , window_shape=(2,2) , strides=(2,2))
        return x , skip

# Upsample Block
# Consists of a Convolutional Transpose, Double Convolution, and Cross Attention
class UpBlock(nnx.Module):
    def __init__(self , in_channels , skip_channels , out_channels , time_embed_dim , context_dim , use_attn , * , rngs):
        self.upsample = nnx.ConvTranspose(in_channels , in_channels , kernel_size=(2,2) , strides=(2,2) , padding='SAME' , rngs=rngs)
        self.double_conv = DoubleConv(in_channels + skip_channels , out_channels , time_embed_dim , rngs=rngs)
        self.attention = CrossAttention(out_channels , context_dim , rngs=rngs) if use_attn else None

    def __call__(self , x , skip , t , context):
        x = self.upsample(x)
        x = jnp.concatenate([x , skip] , axis=-1)
        x = self.double_conv(x , t)
        if self.attention:
            x = self.attention(x , context)
        return x

# Full UNet Model
class UNet(nnx.Module):
    def __init__(self , in_channels , block_out_channels=[128,256,512,512] , time_embed_dim=256 , context_dim=768 , attention_levels=[False,True,True,True] , * , rngs):
        self.time_mlp = nnx.Sequential(
            nnx.Linear(time_embed_dim , time_embed_dim*4 , rngs=rngs),
            nnx.Linear(time_embed_dim*4 , time_embed_dim , rngs=rngs)
        )

        self.init_conv = nnx.Conv(in_channels , block_out_channels[0] , kernel_size=(3,3) , padding='SAME' , rngs=rngs)

        # UNet Encoder
        self.down_blocks = []
        in_channel = block_out_channels[0]
        for i,out_channel in enumerate(block_out_channels):
            self.down_blocks.append(
                DownBlock(in_channel , out_channel , time_embed_dim , context_dim , attention_levels[i] , rngs=rngs)
            )
            in_channel = out_channel
        channel = in_channel

        # UNet Bottleneck
        self.bottleneck = DoubleConv(channel , channel , time_embed_dim , rngs=rngs)
        self.bottleneck_attention = CrossAttention(channel , context_dim , rngs=rngs)

        # UNet Decoder
        self.up_blocks = []
        reversed_channels = list(reversed(block_out_channels))
        reversed_attention = list(reversed(attention_levels))
        for i in range(len(reversed_channels)-1):
            in_channel = reversed_channels[i]
            skip_channel = reversed_channels[i]
            out_channel = reversed_channels[i+1]
            self.up_blocks.append(
                UpBlock(in_channel , skip_channel , out_channel , time_embed_dim , context_dim , reversed_attention[i] , rngs=rngs)
            )
            channel = out_channel
        self.final_up = UpBlock(channel , block_out_channels[i] , channel , time_embed_dim , context_dim , reversed_attention[0] , rngs=rngs)

        # Final projections
        self.final_conv = nnx.Conv(channel , out_channels , kernel_size=(3,3) , padding='SAME' , rngs=rngs)
        self.final_norm = nnx.GroupNorm(num_groups=32 , num_features=out_channels , rngs=rngs)

    def __call__(self , x , t_embed , context):
        t = get_timestep_embedding(t_embed)
        t = self.time_mlp.layers[0](t)
        t = nnx.silu(t)
        t = self.time_mlp.layers[1](t)

        # Initial conv
        x = self.init_conv(x)

        # Encoder
        skips = []
        for block in self.down_blocks:
            x , skip = block(x , t , context)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck(x , t)
        x = self.bottleneck_attention(x , context)

        # Decoder
        for block in self.up_blocks:
            skip = skips.pop()
            x = block(x , skip , t , context)
        skip = skips.pop()
        x = self.final_up(x , skip , t , context)

        x = self.final_norm(x)
        x = nnx.silu(x)
        x = self.final_conv(x)
        return x

