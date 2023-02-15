import sys

import torch
from torch import nn
from torch.nn import functional as F


class EncodeLayer(nn.Module):
    """
        Encoding Layers
    """

    def __init__(self,in_channels,out_channels):
        super(EncodeLayer,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,out_channels,2,stride=2)
        self.pool = nn.MaxPool2d(2,2)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        return x

class DecodeLayer(nn.Module):
    """
        Decoding Layers
    """

    def __init__(self,in_channels,out_channels):
        super(DecodeLayer,self).__init__()

        self.Tconv1 = nn.ConvTranspose2d(in_channels,out_channels,3,padding=1)

    def forward(self,x,SigmAct=0):
        if SigmAct:
            x = torch.sigmoid(self.Tconv1(x))
        else:
            x = F.relu(self.Tconv1(x))

        return x
