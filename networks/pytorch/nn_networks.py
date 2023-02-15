import sys

import torch
from torch import nn
from torch.nn import functional as F

from networks.pytorch.nn_utils import EncodeLayer, DecodeLayer


class FcNet(nn.Module):
    def __init__(self,input_shape,output_shape=7):
        super(FcNet,self).__init__()

        self.fc1 = nn.Linear(input_shape, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, output_shape)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# class DynamicFcNet(nn.Module):
#     def __init__(self,input_shape,output_shape,nLayers=4,nNodes=[200,100,50]):
#         super(DynamicFcNet,self).__init__()
#
#         self.layers = []
#
#         for layer in
#
#     def forward(self,x):
#
#         x = x
#
#         return x

class DeepConvAENet(nn.Module):
    def __init__(self,in_channels,out_channels, filters = [8,32]):
        super(DeepConvAENet,self).__init__()

        assert len(filters) >1, 'Network must contain at least 2 layers'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filters = filters

        encoding = []
        encoding.append(EncodeLayer(self.in_channels,self.filters[-1]))
        encoding.extend([EncodeLayer(self.filters[i+1],self.filters[i]) for i in reversed(range(len(filters)-1))])

        self.encoder_layers = nn.Sequential(*encoding)

        decoding = []
        decoding.extend([DecodeLayer(self.filters[i],self.filters[i+1]) for i in range(len(filters)-1)])

        self.decoder_layers = nn.Sequential(*decoding)


        self.outc = DecodeLayer(self.filters[-1],self.out_channels)

    def forward(self,x):

        x_enc = self.encoder_layers(x)
        x_dec = self.decoder_layers(x_enc)

        logits = self.outc(x_dec,True)
        return logits


class DeepConvNet(nn.Module):
    def __init__(self,in_channels,out_channels, filters= [8,32]):
        super(DeepConvNet,self).__init__()

        assert len(filters) > 1, "Network must contain a few layers.."

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filters = filters

        encoding = []
        encoding.append(EncodeLayer(self.in_channels,self.filters[-1]))
        encoding.extend([EncodeLayer(self.filters[i+1],self.filters[i]) for i in reversed(range(len(filters)-1))])

        self.encoder_layers = nn.Sequential(*encoding)

        self.fc = nn.Linear(filters[0], self.out_channels)

    def forward(self, x):
        x_enc = self.encoder_layers(x)

        x = self.fc(x_enc.flatten())


        return x


class DeepConvSimpleNet(nn.Module):
    """
        Use this net for the 3xN + 1xN vector cluster coords + row number
    """
    def __init__(self,in_channels,out_channels,filter):
        super(DeepConvSimpleNet,self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter = filter

        self.conv1 = EncodeLayer(self.in_channels,self.filter)
        self.fc = nn.Linear(self.filter*2, self.out_channels)

        self.fc11 = nn.Linear(7, 16)
        self.fc12 = nn.Linear(16, out_channels)


    def forward(self,x_3vec,xmP):
        x_3vec = self.conv1(x_3vec)
        x_out1 = self.fc(x_3vec.flatten())

        xmP = F.relu(self.fc11(xmP))
        xmP = self.fc12(xmP)

        return x_out1+xmP
