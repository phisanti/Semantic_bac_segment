import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    """
    Double Convolution Block.

    This class represents a double convolution block in the UNet architecture.
    It consists of two convolutional layers, each followed by batch normalization and ReLU activation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.

    Attributes:
        conv (nn.Sequential): Sequential container of convolutional layers.

    """

    def __init__(self, in_channels, out_channels, dropout_rate=0.2):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        return self.conv(x)


class HEDUNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        features=[64, 128, 256, 512],
        init_features=64,
        pooling_steps=4,
        dropout_rate=0.2,
        deep_supervision=True,
    ):
        super(HEDUNet, self).__init__()

        if features is None:
            features = [2**i for i in range(pooling_steps) if 2**i >= init_features]
        else:
            features = features
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNet
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature, dropout_rate))
            in_channels = feature

        # Up part of UNet
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2,
                    feature,
                    kernel_size=2,
                    stride=2,
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        # HED-like predictors
        self.predictors = nn.ModuleList(
            [
                nn.Conv2d(feature, out_channels, kernel_size=1)
                if i != len(features) - 1
                else nn.Conv2d(feature * 2, out_channels, kernel_size=1)
                for i, feature in enumerate(reversed(features))
            ]
        )

        self.deep_supervision = deep_supervision

    def forward(self, x):
        skip_connections = []
        predictions_list = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        # Apply the last predictor to the output of the bottleneck layer
        predictions_list.append(self.predictors[0](x))

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[-(idx // 2 + 1)]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:], mode="nearest")

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

            if idx < len(self.ups) - 2:
                predictions_list.append(self.predictors[idx // 2 + 1](x))

        final_prediction = self.final_conv(x)
        predictions_list.append(final_prediction)

        if self.deep_supervision:
            return final_prediction, list(reversed(predictions_list))
        else:
            return final_prediction
