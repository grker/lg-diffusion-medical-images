import torch.nn as nn
import torch

from src.utils.hydra_config import StartBlockConfig, ResBlockConfig


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=(3,3), padding='same', bias=True, activation=True):
        super(ConvBlock, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        if activation:
            self.activation = nn.ReLU()
        else:
            self.activation = None
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

    def forward(self, x):
        x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        x = self.conv(x)
        return x
    

class StartBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(StartBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, kernel_size=(3,3), padding="same", bias=True)
        self.convBlock = ConvBlock(out_channels, out_channels, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.convBlock(x)
        return x
    

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, encode=True):
        super(ResBlock, self).__init__()
        self.convBlock1 = ConvBlock(in_channels, out_channels, 2 if encode else 1) 
        self.convBlock2 = ConvBlock(out_channels, out_channels, 1)
        self.identity = ConvBlock(in_channels, out_channels, kernel_size=(1,1), stride=2)
        self.batchNorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = self.identity(x)
        identity = self.batchNorm(identity)

        x = self.convBlock1(x)
        x = self.convBlock2(x)
        x += identity
        return x


class UpsampleBlock(nn.Module):
    def __init__(self):
        super(UpsampleBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=(2,2), mode='nearest')

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        return x
    
class DecodeBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecodeBlock, self).__init__()
        self.upsampleBlock = UpsampleBlock()
        self.resBlock = ResBlock(in_channels, out_channels, encode=False)

    def forward(self, x, skip):
        x = self.upsampleBlock(x, skip)
        x = self.resBlock(x)
        return x