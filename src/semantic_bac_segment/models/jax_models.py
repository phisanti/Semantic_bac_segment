import semantic_bac_segment.models.jax_models as jax_models
import jax.numpy as jnp
import flax.linen as nn

class UNetBlock(nn.Module):
    """UNet Block with convolutional layers, batch normalization, ReLU activation, and dropout."""
    features: int
    dropout_rate: float = 0.2

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=self.features, kernel_size=(3, 3), padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not self.is_mutable_collection('batch_stats'))(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.features, kernel_size=(3, 3), padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not self.is_mutable_collection('batch_stats'))(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not self.is_mutable_collection('dropout'))
        return x

class UNet(nn.Module):
    """UNet for image segmentation with contracting and expanding paths."""
    in_channels: int = 1
    out_channels: int = 1
    init_features: int = 64
    pooling_steps: int = 2
    dropout_rate: float = 0.2

    @nn.compact
    def __call__(self, x):
        assert x.ndim == 4, "Input should be a 4D tensor (batch, height, width, channels)"

        features = self.init_features
        encs = []

        for i in range(self.pooling_steps):
            x = UNetBlock(features=features * (2**i), dropout_rate=self.dropout_rate, name=f"enc{i+1}")(x)
            encs.append(x)
            x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = UNetBlock(features=features * (2**self.pooling_steps), dropout_rate=self.dropout_rate, name="bottleneck")(x)

        for i in reversed(range(self.pooling_steps)):
            x = nn.ConvTranspose(features=features * (2**i), kernel_size=(2, 2), strides=(2, 2))(x)
            x = jnp.concatenate((x, encs[i]), axis=-1)
            x = UNetBlock(features=features * (2**i), dropout_rate=self.dropout_rate, name=f"dec{i+1}")(x)

        x = nn.Conv(features=self.out_channels, kernel_size=(1, 1))(x)
        return nn.sigmoid(x)
