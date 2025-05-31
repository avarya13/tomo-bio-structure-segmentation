import torch
import torch.nn as nn
import torch.nn.functional as F

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


class Encoder(nn.Module):
    """
    An encoder block that applies convolutional layers followed by max pooling.

    Attributes:
        conv (Conv): The convolutional block.
        pool (nn.MaxPool2d): Max pooling layer.
    """

    def __init__(self, in_channels, out_channels):
        """
        Initialize the encoder block.

        Parameters:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """

        super().__init__()
        self.conv = Conv(in_channels, out_channels)
        self.pool = nn.MaxPool2d((2, 2), stride=2)

    def forward(self, x):
        """
        Forward pass through the encoder block.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: The output tensor after convolution and the pooled tensor.
        """

        x = self.conv(x)  
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
        x = F.interpolate(x, size=skip_x.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip_x], axis=1)  
        x = self.conv(x)  
        return x


class Unet(nn.Module):
    def __init__(self, input_channels, num_classes):
        super().__init__()
        self.activations = None

        self.e1 = Encoder(input_channels, 64)
        self.e2 = Encoder(64, 128)
        self.e3 = Encoder(128, 256)
        self.e4 = Encoder(256, 512)

        self.bottle_neck = Conv(512, 1024)
        
        self.d1 = Decoder(1024, 512)
        self.d2 = Decoder(512, 256)
        self.d3 = Decoder(256, 128)
        self.d4 = Decoder(128, 64)
        self.outputs = nn.Conv2d(64, num_classes, kernel_size=1, padding=0)

    def forward(self, x):
        skip_x1, pool_x1 = self.e1(x)
        skip_x2, pool_x2 = self.e2(pool_x1)
        skip_x3, pool_x3 = self.e3(pool_x2)
        skip_x4, pool_x4 = self.e4(pool_x3)

        bottle_neck = self.bottle_neck(pool_x4)

        self.activations = bottle_neck

        d1 = self.d1(bottle_neck, skip_x4)
        d2 = self.d2(d1, skip_x3)
        d3 = self.d3(d2, skip_x2)
        d4 = self.d4(d3, skip_x1)

        outputs = self.outputs(d4)
        return outputs