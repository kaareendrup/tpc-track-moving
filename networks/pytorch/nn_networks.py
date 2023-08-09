import sys

import torch
from torch import nn
from torch.nn import functional as F

from networks.pytorch.nn_utils import EncodeLayer, DecodeLayer


# class FcNet(nn.Module):
#     def __init__(self,input_shape,output_shape=7):
#         super(FcNet,self).__init__()

#         self.fc1 = nn.Linear(input_shape, 200)
#         self.fc2 = nn.Linear(200, 100)
#         self.fc3 = nn.Linear(100, 50)
#         self.fc4 = nn.Linear(50, output_shape)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = self.fc4(x)
#         return x
        
class FcNet(nn.Module):
    def __init__(self,input_shape,output_shape=7):
        super(FcNet,self).__init__()

        self.fc1 = nn.Linear(input_shape, 50)
        self.fc2 = nn.Linear(50, 25)
        self.fc3 = nn.Linear(25, output_shape)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
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
        x_out1 = self.fc(x_3vec.reshape(x_3vec.shape[0],-1))

        xmP = F.relu(self.fc11(xmP))
        xmP = self.fc12(xmP)

        return x_out1+xmP


class PseudoGraph(nn.Module):

    def __init__(self, input_shape, output_shape=7):
        super(PseudoGraph, self).__init__()

        # 7 for the track vector, 6 (5) for the xyz (+3, changed from 2) coordinate of each cluster
        self._n_node_coor = 6
        self._n_track_params = 7

        in_shape_2D = self._n_track_params + self._n_node_coor
        n_clusters = int((input_shape - self._n_track_params) / self._n_node_coor)

        self.fc1 = nn.Linear(in_shape_2D, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 64)
        self.fc3_5 = nn.Linear(64, 32)

        self.fc4 = nn.Linear(n_clusters*32, output_shape)

        # Create dropouts
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.dropout3 = nn.Dropout(0.25)
        self.dropout4 = nn.Dropout(0.25)

        # Create batchnorms. Param is number of input channels = clusters
        self.norm1 = nn.BatchNorm1d(n_clusters)
        self.norm2 = nn.BatchNorm1d(n_clusters)
        self.norm3 = nn.BatchNorm1d(n_clusters)

    def forward(self, x):

        torch.set_printoptions(threshold=10_000)

        x_graph = x[:,self._n_track_params:]
        x_graph = torch.reshape(x_graph, (x_graph.size()[0], -1, self._n_node_coor))

        x_vec = x[:,:self._n_track_params]
        x_repeat = torch.repeat_interleave(x_vec.unsqueeze(1), x_graph.size()[1], 1)
        x = torch.cat([x_graph, x_repeat], dim=2)
        x = F.leaky_relu(self.norm1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.leaky_relu(self.norm2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.leaky_relu(self.norm3(self.fc3(x)))
        x = self.dropout3(x)
        x = F.leaky_relu(self.fc3_5(x))
        x = self.dropout4(x)

        x = nn.Flatten()(x)

        x = self.fc4(x)

        x[:,2] = torch.sigmoid(x[:,2]) * 2 - 1
        x[:,-2] = F.relu(x[:,-2])
        x[:,-1] = F.relu(x[:,-1])

        return x


class PseudoGraphSingleLinear(PseudoGraph):

    def __init__(self, input_shape, output_shape=7):
        super().__init__(input_shape, output_shape)

    def forward(self, x):

        torch.set_printoptions(threshold=10_000)

        x_graph = x[:,self._n_track_params:]
        x_graph = torch.reshape(x_graph, (x_graph.size()[0], -1, self._n_node_coor))

        x_vec = x[:,:self._n_track_params]
        x_repeat = torch.repeat_interleave(x_vec.unsqueeze(1), x_graph.size()[1], 1)
        x = torch.cat([x_graph, x_repeat], dim=2)
        x = F.leaky_relu(self.norm1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.leaky_relu(self.norm2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.leaky_relu(self.norm3(self.fc3(x)))
        x = self.dropout3(x)
        x = F.leaky_relu(self.fc3_5(x))
        x = self.dropout4(x)

        x = nn.Flatten()(x)

        x = self.fc4(x)


class PseudoGraphSinglePhi(PseudoGraph):

    def __init__(self, input_shape, output_shape=7):
        super().__init__(input_shape, output_shape)

    def forward(self, x):

        torch.set_printoptions(threshold=10_000)

        x_graph = x[:,self._n_track_params:]
        x_graph = torch.reshape(x_graph, (x_graph.size()[0], -1, self._n_node_coor))

        x_vec = x[:,:self._n_track_params]
        x_repeat = torch.repeat_interleave(x_vec.unsqueeze(1), x_graph.size()[1], 1)
        x = torch.cat([x_graph, x_repeat], dim=2)
        x = F.leaky_relu(self.norm1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.leaky_relu(self.norm2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.leaky_relu(self.norm3(self.fc3(x)))
        x = self.dropout3(x)
        x = F.leaky_relu(self.fc3_5(x))
        x = self.dropout4(x)

        x = nn.Flatten()(x)

        x = self.fc4(x)

        x[:,0] = torch.sigmoid(x[:,0]) * 2 - 1
        x[:,-1] = F.relu(x[:,-1])

        return x


class PseudoGraphSingleLambda(PseudoGraph):

    def __init__(self, input_shape, output_shape=7):
        super().__init__(input_shape, output_shape)

    def forward(self, x):

        torch.set_printoptions(threshold=10_000)

        x_graph = x[:,self._n_track_params:]
        x_graph = torch.reshape(x_graph, (x_graph.size()[0], -1, self._n_node_coor))

        x_vec = x[:,:self._n_track_params]
        x_repeat = torch.repeat_interleave(x_vec.unsqueeze(1), x_graph.size()[1], 1)
        x = torch.cat([x_graph, x_repeat], dim=2)
        x = F.leaky_relu(self.norm1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.leaky_relu(self.norm2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.leaky_relu(self.norm3(self.fc3(x)))
        x = self.dropout3(x)
        x = F.leaky_relu(self.fc3_5(x))
        x = self.dropout4(x)

        x = nn.Flatten()(x)

        x = self.fc4(x)

        # x[:,0] = torch.sigmoid(x[:,0]) * 2 - 1
        x[:,-1] = F.relu(x[:,-1])

        return x


class mRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(mRNN, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)   
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
    
    def forward(self, x,device):
        x = x.to(device)

        batch_size = x.size(0)

        hidden = self.init_hidden(batch_size)
        hidden = hidden.to(device)

        out, hidden = self.rnn(x, hidden)
        
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        
        return out, hidden
    
    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden