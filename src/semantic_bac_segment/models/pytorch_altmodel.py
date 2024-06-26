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


class SobelLayer(nn.Module):
    def __init__(self):
        super(SobelLayer, self).__init__()
        self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)

        sobel_kernel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        )
        sobel_kernel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
        )

        self.sobel_x.weight = nn.Parameter(
            sobel_kernel_x.float().unsqueeze(0).unsqueeze(0)
        )
        self.sobel_y.weight = nn.Parameter(
            sobel_kernel_y.float().unsqueeze(0).unsqueeze(0)
        )

    def forward(self, x):
        edge_x = self.sobel_x(x)
        edge_y = self.sobel_y(x)
        edges = torch.sqrt(torch.pow(edge_x, 2) + torch.pow(edge_y, 2))
        return edges


class UNET(nn.Module):
    """
    UNet Architecture.

    This class represents the UNet architecture for semantic segmentation.
    It consists of an encoder (contracting path) and a decoder (expanding path) with skip connections.

    Args:
        in_channels (int): Number of input channels (default: 1).
        out_channels (int): Number of output channels (default: 1).
        features (list): List of feature sizes for each level of the UNet (default: [64, 128, 256, 512]).
        init_features (int): Initial number of features (default: 64).
        pooling_steps (int): Number of pooling steps in the encoder (default: 4).

    Attributes:
        ups (nn.ModuleList): List of modules for the expanding path.
        downs (nn.ModuleList): List of modules for the contracting path.
        pool (nn.MaxPool2d): Max pooling layer for downsampling.
        bottleneck (DoubleConv): Bottleneck layer at the bottom of the UNet.
        final_conv (nn.Conv2d): Final convolutional layer for output.

    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        features=[64, 128, 256, 512],
        init_features=64,
        pooling_steps=4,
        dropout_rate=0.2,
    ):
        super(UNET, self).__init__()

        if features == None:
            features = [2**i for i in range(pooling_steps) if 2**i >= init_features]
        else:
            features = features
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature, dropout_rate))
            in_channels = feature

        # Up part of UNET
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
        """Forward pass of the model.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor after passing through the model.
        """
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)


class DualPathwayUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[16, 32, 64], **kwargs):
        super(DualPathwayUNet, self).__init__()

        self.unet1 = UNET(in_channels, out_channels, features, **kwargs)
        self.unet2 = UNET(in_channels, out_channels, features, **kwargs)

    def forward(self, x):
        # Pathway 1: Original image
        out1 = self.unet1(x)

        # Pathway 2: Inverted image
        x_inverted = torch.max(x) - x
        out2 = self.unet2(x_inverted)

        # Merge outputs
        out = out1 + out2

        return torch.sigmoid(out)


class UNET_edges(nn.Module):
    """
    UNet Architecture.

    This class represents the UNet architecture for semantic segmentation.
    It consists of an encoder (contracting path) and a decoder (expanding path) with skip connections.

    Args:
        in_channels (int): Number of input channels (default: 1).
        out_channels (int): Number of output channels (default: 1).
        features (list): List of feature sizes for each level of the UNet (default: [64, 128, 256, 512]).
        init_features (int): Initial number of features (default: 64).
        pooling_steps (int): Number of pooling steps in the encoder (default: 4).

    Attributes:
        ups (nn.ModuleList): List of modules for the expanding path.
        downs (nn.ModuleList): List of modules for the contracting path.
        pool (nn.MaxPool2d): Max pooling layer for downsampling.
        bottleneck (DoubleConv): Bottleneck layer at the bottom of the UNet.
        final_conv (nn.Conv2d): Final convolutional layer for output.

    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        features=[64, 128, 256, 512],
        init_features=64,
        pooling_steps=4,
        dropout_rate=0.2,
    ):
        super(UNET_edges, self).__init__()

        if features == None:
            features = [2**i for i in range(pooling_steps) if 2**i >= init_features]
        else:
            features = features
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.sobel = SobelLayer()

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature, dropout_rate))
            in_channels = feature

        # Up part of UNET
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
        x = self.sobel(x)

        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            x = torch.cat([x, skip_connection], dim=1)  # Fuse features
            x = self.ups[idx + 1](x)

        return torch.sigmoid(self.final_conv(x))
