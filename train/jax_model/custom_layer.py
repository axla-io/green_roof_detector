import jax.numpy as jnp
from flax import linen as nn

class SelfAttentionBlock(nn.Module):
    num_heads: int
    qkv_features: int

    @nn.compact
    def __call__(self, x):
        b, h, w, c = x.shape
        x_flat = x.reshape((b, h * w, c))
        x_attn = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.qkv_features,
            broadcast_dropout=False,
            deterministic=True
        )(x_flat)
        return x_attn.reshape((b, h, w, c))
