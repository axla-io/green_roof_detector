import jax.numpy as jnp
from flax import linen as nn
from custom_layer import SelfAttentionBlock

class ConvBlock(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.features, (3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(self.features, (3, 3), padding='SAME')(x)
        x = nn.relu(x)
        return x

class UNetWithAttention(nn.Module):
    base_features: int = 64

    @nn.compact
    def __call__(self, x):
        # Encoder
        x1 = ConvBlock(self.base_features)(x)
        x2 = nn.max_pool(x1, (2, 2))

        x2 = ConvBlock(self.base_features * 2)(x2)
        x3 = nn.max_pool(x2, (2, 2))

        # Bottleneck with self-attention only
        x3 = ConvBlock(self.base_features * 4)(x3)
        x3 = SelfAttentionBlock(num_heads=4, qkv_features=self.base_features * 4)(x3)

        # Decoder
        x = nn.ConvTranspose(self.base_features * 2, (2, 2), strides=(2, 2))(x3)
        x = jnp.concatenate([x, x2], axis=-1)
        x = ConvBlock(self.base_features * 2)(x)

        x = nn.ConvTranspose(self.base_features, (2, 2), strides=(2, 2))(x)
        x = jnp.concatenate([x, x1], axis=-1)
        x = ConvBlock(self.base_features)(x)

        # Output layer
        x = nn.Conv(1, (1, 1))(x)
        return nn.sigmoid(x)  # for binary segmentation
