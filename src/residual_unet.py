import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    A residual block consisting of two convolutional layers with batch normalization and a shortcut connection.

    Attributes:
        conv1 (nn.Conv2d): The first convolutional layer.
        bn1 (nn.BatchNorm2d): Batch normalization after the first convolution.
        conv2 (nn.Conv2d): The second convolutional layer.
        bn2 (nn.BatchNorm2d): Batch normalization after the second convolution.
        shortcut (nn.Sequential): A shortcut connection to match dimensions if needed.
    """

    def __init__(self, in_channels, out_channels):
        """
        Initialize the residual block.

        Parameters:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """

        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        """
        Forward pass through the residual block.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after residual connection and ReLU activation.
        """

        residual = self.shortcut(x)  
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += residual
        return F.relu(x)

class Conv(nn.Module):
    """
    A convolutional block consisting of two convolutional layers followed by batch normalization.

    Attributes:
        conv1 (nn.Conv2d): The first convolutional layer.
        bn1 (nn.BatchNorm2d): Batch normalization after the first convolution.
        conv2 (nn.Conv2d): The second convolutional layer.
        bn2 (nn.BatchNorm2d): Batch normalization after the second convolution.
    """

    def __init__(self, in_channels, out_channels):
        """
        Initialize the convolutional block.

        Parameters:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """

        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        """
        Forward pass through the convolutional block.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after convolution and ReLU activation.
        """

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return F.relu(x)
    

class SharedChannelConv(nn.Module):
    """
    A convolutional layer that applies the same 3x3 convolution across all channels,
    followed by a 1x1 convolution to combine the outputs or by using min/max aggregation.
    """

    def __init__(self, in_channels, out_channels, channels_aggr='conv'):
        super().__init__()
        # Define a single 3x3 convolution that will be applied across all channels
        self.shared_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        
        # Define aggregation method based on input parameter
        self.channels_aggr = channels_aggr
        if self.channels_aggr == 'conv':
            self.pointwise_conv = nn.Conv2d(in_channels, 1, kernel_size=1)

        self.final_conv = nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        in_channels = x.shape[1]
        
        # Apply shared 3x3 convolution to each channel independently
        out = []
        for i in range(in_channels):
            out.append(self.shared_conv(x[:, i:i+1, :, :]))  

        out = torch.cat(out, dim=1)
        
        # Perform aggregation based on specified method
        if self.channels_aggr == 'conv':
            x = self.pointwise_conv(out)
            x = F.relu(x)
        elif self.channels_aggr == 'max':
            x = torch.max(out, dim=1).values
            x = x.unsqueeze(1)   
        elif self.channels_aggr == 'min':
            x = torch.min(out, dim=1).values  
            x = x.unsqueeze(1) 
        else:
            raise ValueError("Invalid aggregation method. Use 'conv', 'max', or 'min'.")
        return self.final_conv(x)

class Encoder(nn.Module):
    """
    An encoder block that applies a convolutional block followed by multiple residual blocks and max pooling.

    Attributes:
        conv (Conv): The convolutional block.
        residual_blocks (nn.Sequential): Sequential application of residual blocks.
        pool (nn.MaxPool2d): Max pooling layer.
    """

    def __init__(self, in_channels, out_channels, num_blocks, channels_aggr=None, use_shared_conv=False):
        """
        Initialize the encoder block.

        Parameters:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            num_blocks (int): Number of residual blocks to apply.
        """

        super().__init__()
        #self.conv = Conv(in_channels, out_channels)
        self.conv = SharedChannelConv(in_channels, out_channels, channels_aggr) if use_shared_conv else Conv(in_channels, out_channels)
        self.residual_blocks = self.make_residual(out_channels, num_blocks)
        self.pool = nn.MaxPool2d((2, 2), stride=2)

    def make_residual(self, out_channels, num_blocks):
        """
        Create a sequential container of residual blocks.

        Parameters:
            out_channels (int): Number of output channels for the residual blocks.
            num_blocks (int): Number of residual blocks to create.

        Returns:
            nn.Sequential: A sequential container of residual blocks.
        """

        layers = []
        for _ in range(num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the encoder block.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: The output tensor after convolutions and the pooled tensor.
        """

        x = self.conv(x)  
        x = self.residual_blocks(x)
        p = self.pool(x)
        return x, p

class Decoder(nn.Module):
    """
    A decoder block that upsamples the input and concatenates it with the skip connection from the encoder.

    Attributes:
        up (nn.ConvTranspose2d): Transposed convolutional layer for upsampling.
        conv (Conv): The convolutional block.
    """

    def __init__(self, in_channels, out_channels):
        """
        Initialize the decoder block.

        Parameters:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """

        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        self.conv = Conv(out_channels * 2, out_channels)

    def forward(self, x, skip_x):
        """
        Forward pass through the decoder block.

        Parameters:
            x (torch.Tensor): Input tensor from the previous layer.
            skip_x (torch.Tensor): Skip connection tensor from the encoder.

        Returns:
            torch.Tensor: Output tensor after upsampling and convolution.
        """

        x = self.up(x)
        # skip_x = F.interpolate(skip_x, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
        x = torch.cat([x, skip_x], axis=1) 
        x = self.conv(x)
        return x

class ResidualUnet(nn.Module):
    """
    Residual U-Net architecture for image segmentation.

    Attributes:
        e1 (Encoder): First encoder block.
        e2 (Encoder): Second encoder block.
        e3 (Encoder): Third encoder block.
        e4 (Encoder): Forth encoder block.
        bottle_conv (nn.Conv2d): Bottleneck convolutional layer.
        bottle_bn (nn.BatchNorm2d): Batch normalization after bottleneck convolution.
        bottle_residual (ResidualBlock): Residual block after bottleneck convolution.
        d1 (Decoder): First decoder block.
        d2 (Decoder): Second decoder block.
        d3 (Decoder): Third decoder block.
        d4 (Decoder): Forth decoder block.
        outputs (nn.Conv2d): Final convolutional layer to produce output segmentation map.
    """

    def __init__(self, input_channels, num_classes, channels_aggr='conv'):
        """
        Initialize the Residual U-Net model.

        Parameters:
            input_channels (int): Number of input channels.
            num_classes (int): Number of output classes for segmentation.
        """

        super().__init__()
        self.activations = None
        use_shared_conv = True if input_channels > 1 else False 

        self.e1 = Encoder(input_channels, 16, 1, channels_aggr, use_shared_conv)
        self.e2 = Encoder(16, 32, 1)
        self.e3 = Encoder(32, 64, 1)
        self.e4 = Encoder(64, 128, 1)
        
        self.bottle_conv = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bottle_bn = nn.BatchNorm2d(256)
        self.bottle_residual = ResidualBlock(256, 256)
        
        self.d1 = Decoder(256, 128)
        self.d2 = Decoder(128, 64)
        self.d3 = Decoder(64, 32)
        self.d4 = Decoder(32, 16)
        self.outputs = nn.Conv2d(16, num_classes, kernel_size=1, padding=0)

    def forward(self, x):
        """
        Forward pass through the Residual U-Net model.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output segmentation map.
        """
        skip_x1, pool_x1 = self.e1(x)
        skip_x2, pool_x2 = self.e2(pool_x1)
        skip_x3, pool_x3 = self.e3(pool_x2)
        skip_x4, pool_x4 = self.e4(pool_x3)

        bottle_neck = F.relu(self.bottle_bn(self.bottle_conv(pool_x4)))
        bottle_neck = self.bottle_residual(bottle_neck)

        self.activations = bottle_neck

        d1 = self.d1(bottle_neck, skip_x4)
        d2 = self.d2(d1, skip_x3)
        d3 = self.d3(d2, skip_x2)
        d4 = self.d4(d3, skip_x1)

        outputs = self.outputs(d4)
        return outputs
