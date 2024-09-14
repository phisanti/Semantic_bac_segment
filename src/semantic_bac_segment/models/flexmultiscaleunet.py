import torch
import torch.nn as nn
from monai.networks.blocks import Convolution
from monai.networks.layers import Act, Norm

class MultiScaleBlock(nn.Module):
    """
    A multi-scale convolutional block that processes input at multiple scales and combines the results.

    This block applies multiple convolutional operations at different scales to the input,
    concatenates the results, and applies a shortcut connection. 

    Args:
        num_in_channels (int): Number of input channels.
        num_filters (int): Base number of filters for the convolutional layers.
        scales (List[float]): List of scales to apply to the number of filters.
        kernel_sizes (Union[int, List[int]]): Kernel size(s) for the convolutional layers.
            If an integer is provided, it will be used for all scales.
            If a list is provided, it should have the same length as `scales`.

    Attributes:
        convs (nn.ModuleList): List of convolutional layers for each scale.
        shortcut (Convolution): Shortcut connection to match input and output dimensions.
        batch_norm1 (nn.BatchNorm2d): Batch normalization layer after concatenation.
        batch_norm2 (nn.BatchNorm2d): Batch normalization layer after shortcut addition.

    Forward Pass:
        1. Apply shortcut convolution to input.
        2. Apply each scale's convolution to the input.
        3. Concatenate the outputs from all scales.
        4. Apply batch normalization.
        5. Add the shortcut connection.
        6. Apply another batch normalization.
        7. Apply ReLU activation.

    Returns:
        torch.Tensor: The output tensor after multi-scale processing and shortcut connection.

    Note:
        The total number of output channels is the sum of `num_filters * scale` for all scales.
        This allows the block to maintain a variable output size based on the provided scales.
    """

    def __init__(self, num_in_channels, num_filters, scales, kernel_sizes=3):
        super().__init__()
        self.convs = nn.ModuleList()

        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * len(scales)

        total_out_channels = 0

        for scale, ks in zip(scales, kernel_sizes):

            out_channels = int(num_filters * scale)
            self.convs.append(
                Convolution(
                    spatial_dims=2,
                    in_channels=num_in_channels,  # Use num_in_channels for all convolutions
                    out_channels=out_channels,
                    kernel_size=ks,
                    act=Act.RELU,
                    norm=Norm.BATCH,
                    padding='same',
                    bias=False
                )
            )
            total_out_channels += out_channels
        
        self.shortcut = Convolution(
            spatial_dims=2,
            in_channels=num_in_channels,
            out_channels=total_out_channels,
            kernel_size=1,
            act=None,
            norm=Norm.BATCH,
            padding='same',
            bias=False
        )

        self.batch_norm1 = nn.BatchNorm2d(total_out_channels)
        self.batch_norm2 = nn.BatchNorm2d(total_out_channels)

    def forward(self, x):
        shrtct = self.shortcut(x)
        
        outputs = []
        for conv in self.convs:
            outputs.append(conv(x))  # Apply each conv directly to the input x
        x = torch.cat(outputs, dim=1)
        x = self.batch_norm1(x)

        x = x + shrtct
        x = self.batch_norm2(x)
        x = nn.functional.relu(x)

        return x



class Respath(nn.Module):
    '''
    ResPath
    
    Arguments:
        num_in_filters {int} -- Number of filters going in the respath
        num_out_filters {int} -- Number of filters going out the respath
        respath_length {int} -- length of ResPath
    '''
    def __init__(self, num_in_filters, num_out_filters, respath_length):
        super().__init__()
        self.respath_length = respath_length
        self.shortcuts = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(self.respath_length):
            if i == 0:
                self.shortcuts.append(
                    Convolution(
                        spatial_dims=2,
                        in_channels=num_in_filters,
                        out_channels=num_out_filters,
                        kernel_size=1,
                        act=None,
                        norm=Norm.BATCH,
                        padding='same'
                    )
                )
                self.convs.append(
                    Convolution(
                        spatial_dims=2,
                        in_channels=num_in_filters,
                        out_channels=num_out_filters,
                        kernel_size=3,
                        act=Act.RELU,
                        norm=Norm.BATCH,
                        padding='same'
                    )
                )
            else:
                self.shortcuts.append(
                    Convolution(
                        spatial_dims=2,
                        in_channels=num_out_filters,
                        out_channels=num_out_filters,
                        kernel_size=1,
                        act=None,
                        norm=Norm.BATCH,
                        padding='same'
                    )
                )
                self.convs.append(
                    Convolution(
                        spatial_dims=2,
                        in_channels=num_out_filters,
                        out_channels=num_out_filters,
                        kernel_size=3,
                        act=Act.RELU,
                        norm=Norm.BATCH,
                        padding='same'
                    )
                )
            self.bns.append(nn.BatchNorm2d(num_out_filters))

    def forward(self, x):
        for i in range(self.respath_length):
            shortcut = self.shortcuts[i](x)
            x = self.convs[i](x)
            x = self.bns[i](x)
            x = nn.functional.relu(x)
            x = x + shortcut
            x = self.bns[i](x)
            x = nn.functional.relu(x)
        return x

class FlexMultiScaleUNet(nn.Module):
    def __init__(self, input_channels, num_classes, scales, kernel_sizes=3, base_filters=32, num_steps=4):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.scales = scales
        self.kernel_sizes = kernel_sizes if isinstance(kernel_sizes, list) else [kernel_sizes] * len(scales)
        self.base_filters = base_filters
        self.num_steps = num_steps


        # Encoder Path
        self.encoder_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.respaths = nn.ModuleList()
        
        in_channels = input_channels
        num_filters = base_filters
        encoder_out_channels = []
        
        for i in range(self.num_steps):
            self.encoder_blocks.append(MultiScaleBlock(in_channels, num_filters, scales, kernel_sizes))
            out_channels = self.calculate_out_channels(num_filters)
            encoder_out_channels.append(out_channels)
            self.pools.append(nn.MaxPool2d(2))
            self.respaths.append(Respath(out_channels, out_channels, respath_length=num_steps-i))
            
            in_channels = out_channels  # Update in_channels for the next iteration
            num_filters *= 2

        # Bottleneck layer
        self.bottleneck = MultiScaleBlock(in_channels, num_filters, scales, kernel_sizes)
        bottleneck_out_channels = self.calculate_out_channels(num_filters)

        # Decoder Path
        self.decoder_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        
        decoder_filters = num_filters
        for i in range(self.num_steps):
            # Use bottleneck_out_channels for the first upsample, then use the output of the previous decoder block
            upsample_in_channels = bottleneck_out_channels if i == 0 else decoder_out_channels
            self.upsamples.append(nn.ConvTranspose2d(upsample_in_channels, upsample_in_channels, kernel_size=2, stride=2))
            
            decoder_in_channels = upsample_in_channels + encoder_out_channels[-(i+1)]
            decoder_out_channels = self.calculate_out_channels(num_filters // 2)
            self.decoder_blocks.append(MultiScaleBlock(decoder_in_channels, num_filters // 2, scales, self.kernel_sizes))            
            num_filters //= 2


        # Final convolution layer to produce the output segmentation map
        self.conv_final = Convolution(
            spatial_dims=2,
            in_channels=decoder_out_channels,
            out_channels=num_classes,
            kernel_size=1,
            act=None,
            norm=Norm.BATCH,
            padding='same'
        )

    def calculate_out_channels(self, num_filters):
        return sum(int(num_filters * scale) for scale in self.scales)

    def forward(self, x):
        encoder_outputs = []
        
        # Forward pass through the encoder path
        for i in range(self.num_steps):
            x = self.encoder_blocks[i](x)
            encoder_outputs.append(x)
            x = self.pools[i](x)
            x = self.respaths[i](x)  # Apply Respath to x, not encoder_outputs[-1]

        # Forward pass through the bottleneck layer
        x = self.bottleneck(x)

        # Forward pass through the decoder path
        for i in range(self.num_steps):
            x = self.upsamples[i](x)
            encoder_output = encoder_outputs[-(i+1)]            
            x = torch.cat([x, encoder_outputs[self.num_steps-i-1]], dim=1)
            x = self.decoder_blocks[i](x)


        # Final convolution to produce the output
        out = self.conv_final(x)

        return out


if __name__ == "__main__":

    # Define model parameters
    input_channels = 1
    num_classes = 1
    scales = [1, 0.5, 0.25]
    kernel_sizes=[3, 3, 5]
    base_filters = 32
    num_steps = 2

    # Create the model
    model = FlexMultiScaleUNet(input_channels, num_classes, scales, kernel_sizes, base_filters, num_steps)

    # Create a random input tensor
    input_tensor = torch.randn(1, 1, 32, 32)

    # Run the model
    output = model(input_tensor)

    # Print the output shape
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")