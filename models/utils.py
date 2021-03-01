import torch
import torch.nn as nn



def conv3x3(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)


class ResidualBlock(nn.Module):
    """Simple residual block with two 3x3 convolutions.
    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch)
        if in_ch != out_ch:
            self.skip = conv1x1(in_ch, out_ch)
        else:
            self.skip = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu(out)

        if self.skip is not None:
            identity = self.skip(x)

        out = out + identity
        return out


def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )


def deconv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )


class OffsetBottleneck(JointAutoregressiveHierarchicalPriors):

    def __init__(self, N, M, **kwargs):
        super().__init__(N, M, **kwargs)
        
        conv1 = conv3x3(128 + 18, N)
        conv2 = ResidualBlock(N,N)
        conv3 = ResidualBlock(N,N)
        conv4 = conv3x3(N, M)
        
        deconv1 = conv3x3(M, N)
        deconv2 = ResidualBlock(N,N)
        deconv3 = ResidualBlock(N,N)
        deconv4 = conv3x3(N, 18)
        

        self.g_a = nn.Sequential(
            conv1,
            nn.ReLU(),
            conv2,
            nn.ReLU(),
            conv3,
            nn.ReLU(),
            conv4
        )

        self.g_s = nn.Sequential(
            deconv1,
            nn.ReLU(),
            deconv2,
            nn.ReLU(),
            deconv3,
            nn.ReLU(),
            deconv4
        )



class ResidualBottleneck(JointAutoregressiveHierarchicalPriors):

    def __init__(self, N, M, **kwargs):
        super().__init__(N, M, **kwargs)

        conv1 = conv(3, N)
        conv2 = conv(N, N)
        conv3 = conv(N, N)
        conv4 = conv(N, M)

        deconv1 = deconv(M, N)
        deconv2 = deconv(N, N)
        deconv3 = deconv(N, N)
        deconv4 = deconv(N, 3)


        self.g_a = nn.Sequential(
            conv1,
            nn.ReLU(),
            conv2,
            nn.ReLU(),
            conv3,
            nn.ReLU(),
            conv4
        )

        self.g_s = nn.Sequential(
            deconv1,
            nn.ReLU(),
            deconv2,
            nn.ReLU(),
            deconv3,
            nn.ReLU(),
            deconv4
        )