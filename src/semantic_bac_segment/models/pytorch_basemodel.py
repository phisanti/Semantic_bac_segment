import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Tuple


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        init_features=64,
        pooling_steps=2,
        dropout_rate=0.2,
    ):
        super(UNet, self).__init__()

        features = init_features
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upconv = nn.ModuleList()
        self.dropout_rate = dropout_rate

        for i in range(pooling_steps):
            input_features = in_channels if i == 0 else features * (2 ** (i - 1))
            output_features = features * (2**i)
            self.encoders.append(
                UNet._block(input_features, output_features, name=f"enc{i+1}")
            )
            self.decoders.insert(
                0, UNet._block(output_features * 2, output_features, name=f"dec{i+1}")
            )
            self.upconv.insert(
                0,
                nn.ConvTranspose2d(
                    output_features * 2, output_features, kernel_size=2, stride=2
                ),
            )

        self.bottleneck = UNet._block(
            features * (2 ** (pooling_steps - 1)),
            features * (2**pooling_steps),
            name="bottleneck",
            dropout_rate=self.dropout_rate,
        )
        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def load_weights(self, weights_path):
        """
        Load weights from a previously trained model.

        Args:
            weights_path (str): Path to the weights file.
        """
        self.load_state_dict(torch.load(weights_path))

    @classmethod
    def from_pretrained(cls, weights_path, **kwargs):
        """
        Create a UNet model and initialize it with pretrained weights.

        Args:
            weights_path (str): Path to the pretrained weights file.
            **kwargs: Additional arguments to pass to the constructor.

        Returns:
            UNet: Initialized UNet model with pretrained weights.
        """
        model = cls(**kwargs)
        model.load_weights(weights_path)
        return model

    def forward(self, x):
        encs = []
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            encs.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for i, decoder in enumerate(self.decoders):
            x = self.upconv[i](x)
            x = torch.cat((x, encs[-(i + 1)]), dim=1)
            x = decoder(x)

        return self.conv(x)

    @staticmethod
    def _block(in_channels, features, name, dropout_rate=0.2):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        f"{name}_conv2d",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (f"{name}_norm2d1", nn.BatchNorm2d(num_features=features)),
                    (f"{name}_relu1", nn.ReLU(inplace=True)),
                    (
                        f"{name}_conv2d2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (f"{name}_norm2d2", nn.BatchNorm2d(num_features=features)),
                    (f"{name}_relu2", nn.ReLU(inplace=True)),
                    (f"{name}_dropout", nn.Dropout(p=dropout_rate)),
                ]
            )
        )


class DoubleConvBlock(nn.Module):
    """
    Double convolutional block with batch normalization and dropout.

    Attributes:
        conv1 (nn.Conv2d): First convolutional layer.
        bn1 (nn.BatchNorm2d): Batch normalization layer after the first convolutional layer.
        relu (nn.ReLU): ReLU activation function.
        conv2 (nn.Conv2d): Second convolutional layer.
        bn2 (nn.BatchNorm2d): Batch normalization layer after the second convolutional layer.
        dropout (nn.Dropout): Dropout layer.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        dropout_rate: float = 0.2,
    ):
        super(DoubleConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the double convolutional block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class DoubleComb_UNet(nn.Module):
    """
    U-Net architecture for image segmentation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        init_features: int = 64,
        pooling_steps: int = 4,
        dropout_rate: float = 0.2,
    ):
        super(DoubleComb_UNet, self).__init__()
        self.pooling_steps = pooling_steps
        self.dropout_rate = dropout_rate
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Encoding layers
        self.encoder1 = DoubleConvBlock(in_channels, init_features)
        self.pool1 = nn.MaxPool2d(2)

        self.encoder2 = DoubleConvBlock(init_features, init_features * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.encoder3 = DoubleConvBlock(init_features * 2, init_features * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.encoder4 = DoubleConvBlock(init_features * 4, init_features * 8)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConvBlock(init_features * 8, init_features * 16)

        # Decoding layers
        self.upconv4 = nn.ConvTranspose2d(
            init_features * 16, init_features * 8, 2, stride=2
        )
        self.decoder4 = DoubleConvBlock(init_features * 16, init_features * 8)

        self.upconv3 = nn.ConvTranspose2d(
            init_features * 8, init_features * 4, 2, stride=2
        )
        self.decoder3 = DoubleConvBlock(init_features * 8, init_features * 4)

        self.upconv2 = nn.ConvTranspose2d(
            init_features * 4, init_features * 2, 2, stride=2
        )
        self.decoder2 = DoubleConvBlock(init_features * 4, init_features * 2)

        self.upconv1 = nn.ConvTranspose2d(init_features * 2, init_features, 2, stride=2)
        self.decoder1 = DoubleConvBlock(init_features * 2, init_features)

        self.final = nn.Conv2d(init_features, out_channels, 1)

        # Initialize weights
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the U-Net architecture.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """

        # Encoding
        conv1 = self.encoder1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.encoder2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.encoder3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.encoder4(pool3)
        pool4 = self.pool4(conv4)

        bottleneck = self.bottleneck(pool4)

        # Decoding
        up4 = self.upconv4(bottleneck)
        up4 = torch.cat([up4, conv4], dim=1)
        dec4 = self.decoder4(up4)

        up3 = self.upconv3(dec4)
        up3 = torch.cat([up3, conv3], dim=1)
        dec3 = self.decoder3(up3)

        up2 = self.upconv2(dec3)
        up2 = torch.cat([up2, conv2], dim=1)
        dec2 = self.decoder2(up2)

        up1 = self.upconv1(dec2)
        up1 = torch.cat([up1, conv1], dim=1)
        dec1 = self.decoder1(up1)

        output = self.final(dec1)

        return output

    def _initialize_weights(self) -> None:
        """
        Initialize weights using Xavier/He initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


# Example usage

if __name__ == "__main__":
    model = UNet(
        in_channels=1,
        out_channels=1,
        init_features=64,
        pooling_steps=4,
        dropout_rate=0.2,
    )
    input_tensor = torch.randn(1, 1, 256, 256)
    output = model(input_tensor)
    print(output.shape)
