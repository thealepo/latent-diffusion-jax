'''
unet.py

UNet architecture implemented in Flax NNX.

Architecture overview:
  init_conv
  ↓
  DownBlock x N  (DoubleConv → optional CrossAttention → MaxPool) + skip connections
  ↓
  Bottleneck     (DoubleConv → CrossAttention)
  ↓
  UpBlock x N    (ConvTranspose → concat skip → DoubleConv → optional CrossAttention)
  ↓
  final_norm → silu → final_conv
'''

from typing import List , Optional , Tuple
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from einops import rearrange


# Positional Embedding (Timestep)
def get_timestep_embedding(timesteps: jnp.ndarray , embedding_dim: int = 256) -> jnp.ndarray:
    '''
    Transformer-style positional embedding for timesteps.
    '''
    half_dim = embedding_dim // 2
    exponent = -jnp.log(10000.0) * jnp.arange(half_dim) / half_dim
    emb = jnp.exp(exponent)
    emb = timesteps[: , None] * emb[None , :]
    return jnp.concatenate([jnp.sin(emb) , jnp.cos(emb)] , axis=-1)

# Double Convolution Block
class DoubleConv(nnx.Module):
    def __init__(self , in_channels: int , out_channels: int , time_embed_dim: int , * , rngs: jax.Array) -> None:
        # First convolution
        self.conv1 = nnx.Conv(in_channels , out_channels , kernel_size=(3,3) , padding='SAME' , rngs=rngs)
        self.norm1 = nnx.GroupNorm(num_groups=32 , num_features=out_channels , rngs=rngs)
        # Second convolution
        self.conv2 = nnx.Conv(out_channels , out_channels , kernel_size=(3,3) , padding='SAME' , rngs=rngs)
        self.norm2 = nnx.GroupNorm(num_groups=32 , num_features=out_channels , rngs=rngs)

        # Timestep MLP projection
        self.time_mlp = nnx.Linear(time_embed_dim , out_channels , rngs=rngs)

    def __call__(self , x: jnp.ndarray , time_embed: jnp.ndarray) -> jnp.ndarray:
        # First Convolution
        x = self.conv1(x)
        x = self.norm1(x)
        x = nnx.silu(x)

        # Time embeddings
        t = self.time_mlp(nnx.silu(time_embed))  # (B , out_channels)
        t = rearrange(t , 'b c -> b 1 1 c')      # (B , 1 , 1 , out_channels)
        x = x + t

        # Second Convolution
        x = self.conv2(x)
        x = self.norm2(x)
        x = nnx.silu(x)

        return x

# Cross Attention Block
# Query (Image) attends to Key (Text) and Value (Text)
class CrossAttention(nnx.Module):
    '''
    Multi-headed cross-attention.
    '''
    def __init__(self , channels: int , context_dim: int , n_heads: int = 8 , * , rngs: nnx.RNGs) -> None:
        self.n_heads = n_heads
        self.head_dim = channels // n_heads
        self.norm = nnx.GroupNorm(num_groups=32 , num_features=channels , rngs=rngs)

        # Projection matrices
        self.wq = nnx.Linear(channels , channels , rngs=rngs)
        self.wk = nnx.Linear(context_dim , channels , rngs=rngs)
        self.wv = nnx.Linear(context_dim , channels , rngs=rngs)
        self.wo = nnx.Linear(channels , channels , rngs=rngs)

    def __call__(self , x , context):
        B , H , W , C = x.shape
        residual = x

        # Normalize, and then flatten spatial dimensions (for attention)
        h = self.norm(x)
        h_flat = rearrange(h , 'b h w c -> b (h w) c')  # (B, H*W , C)

        # Compute Q from image, K and V from text
        Q = self.wq(h_flat)    # (B , H*W , C)
        K = self.wk(context)   # (B , seq_len , C)
        V = self.wv(context)   # (B , seq_len , C)

        # Splitting into heads
        def mha_reshape(tensor):
            return rearrange(tensor , 'b n (head d) -> b head n d' , head=self.n_heads)
        Q , K , V = map(mha_reshape , (Q , K , V))

        # Scaled dot product
        scale = 1.0 / jnp.sqrt(self.head_dim)
        attention = jnp.einsum('b h i d , b h j d -> b h i j' , Q , K) * scale
        attention = nnx.softmax(attention , axis=-1)
        out = jnp.einsum('b h i j , b h j d -> b h i d' , attention , V) 

        out = rearrange(out , 'b h n d -> b n (h d)') # merging heads back together
        out = self.wo(out)
        out = out.reshape((B , H , W , C)) # (B , H , W , C)

        return out + residual

# Downsample Block
# Consists of DoubleConv, CrossAttention, followed by a max pool
class DownBlock(nnx.Module):
    '''
    Encoder block: DoubleConv → optional CrossAttention → MaxPool
    Returns both the downsampled feature map and the skip connection.
    '''
    def __init__(self , in_channels: int , out_channels: int , time_embed_dim: int , context_dim: int , use_attn: bool , * , rngs: nnx.RNGs) -> None:
        self.double_conv = DoubleConv(in_channels , out_channels , time_embed_dim , rngs=rngs)
        self.attention = CrossAttention(out_channels , context_dim , rngs=rngs) if use_attn else None

    def __call__(self , x: jnp.ndarray , t: jnp.ndarray , context: jnp.ndarray) -> Tuple[jnp.ndarray , jnp.ndarray]:
        x = self.double_conv(x , t)
        if self.attention:
            x = self.attention(x , context)
        
        skip = x # saving the skip connection
        x = nnx.max_pool(x , window_shape=(2,2) , strides=(2,2))
        return x , skip

# Upsample Block
# Consists of a Convolutional Transpose, Double Convolution, and Cross Attention
class UpBlock(nnx.Module):
    '''
    Decoder block: ConvTranspose → concat skip → DoubleConv → optional CrossAttention
    Returns the upsampled feature map.
    '''
    def __init__(self , in_channels: int , skip_channels: int , out_channels: int , time_embed_dim: int , context_dim: int , use_attn: bool , * , rngs: nnx.RNGs) -> None:
        self.upsample = nnx.ConvTranspose(in_channels , in_channels , kernel_size=(2,2) , strides=(2,2) , padding='SAME' , rngs=rngs)
        self.double_conv = DoubleConv(in_channels + skip_channels , out_channels , time_embed_dim , rngs=rngs)
        self.attention = CrossAttention(out_channels , context_dim , rngs=rngs) if use_attn else None

    def __call__(self , x: jnp.ndarray , skip: jnp.ndarray , t: jnp.ndarray , context: jnp.ndarray) -> jnp.ndarray:
        x = self.upsample(x) # (B, H*2 , W*2 , C)
        x = jnp.concatenate([x , skip] , axis=-1)
        x = self.double_conv(x , t)
        if self.attention:
            x = self.attention(x , context)
        return x

# Full UNet Model
class UNet(nnx.Module):
    '''
    Full UNet.
    Configuration is implemented for 256x256x3 images encoded as 32x32x4 latents.
    '''
    def __init__(self , in_channels: int , block_out_channels: List[int] = [128,256,512,512] , time_embed_dim: int = 256 , context_dim: int = 768 , attention_levels: List[bool] = [False,True,True,True] , * , rngs: nnx.RNGs) -> None:
        # Timestep MLP Projection: embedding_dim -> 4*embedding_dim -> embedding_dim
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
        self.final_conv = nnx.Conv(channel , in_channels , kernel_size=(3,3) , padding='SAME' , rngs=rngs)
        self.final_norm = nnx.GroupNorm(num_groups=32 , num_features=out_channels , rngs=rngs)

    def __call__(self , x: jnp.ndarray , timesteps: jnp.ndarray , context: jnp.ndarray) -> jnp.ndarray:
        # Build the timetep embedding and pass through the MLP
        t = get_timestep_embedding(t_embed)  # (B , time_embed_dim)
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

