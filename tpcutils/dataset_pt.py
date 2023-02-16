import sys,os

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from tpcutils.data import DataHandler,SeparatedDataHandler

#### PYTORCH

class TPCClusterDataset(Dataset):
    def __init__(self, tracks_path, mov_path, transform=False):

        self.X = DataHandler(tracks_path)
        self.y = DataHandler(mov_path)


        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx,:]
        y = self.y[idx,:]

        if self.transform:
            x = self._transform(x)
            y = self._transform(y)

        # x_tensor = torch.tensor(x,dtype=torch.float64)
        # y_tensor = torch.tensor(y,dtype=torch.float64)
        x_tensor = torch.from_numpy(x).float()
        y_tensor = torch.from_numpy(y).float()


        return x_tensor,y_tensor

    def _transform(self,array):
        return (array - array.min())/(array.max()-array.min())

    def _shape(self,):
        return self.X[2,:].shape[0]

class TPCClusterDatasetConvolutional(Dataset):
    def __init__(self, tracks_path, mov_path,nTPCclusters=20, transform=False,tpcNorm=True):

        self.X = SeparatedDataHandler(tracks_path,nTPCclusters)
        self.Y = SeparatedDataHandler(mov_path,nTPCclusters)
        self.y = self.Y['xamP']

        if tpcNorm:
            self.X['xyz'] = (self.X['xyz'] + 260)/(260+260)

        self.x = np.column_stack((self.X['xyz'],self.X['pads'][:,np.newaxis,:]))

        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):

        x = self.x[idx,...]
        xamP = self.X['xamP'][idx,...]



        y = self.y[idx,2:]




        if self.transform:
            x = self._transform(x)
            y = self._transform(y)



        # x_tensor = torch.tensor(x,dtype=torch.float64)
        # y_tensor = torch.tensor(y,dtype=torch.float64)
        x_tensor = torch.from_numpy(x).float().unsqueeze(0)
        y_tensor = torch.from_numpy(y).float()

        mP_tensor = torch.from_numpy(xamP).float()


        tensor_data = {}
        tensor_data['input_xyz_row'] = x_tensor
        tensor_data['mP'] = mP_tensor
        tensor_data['target'] = y_tensor

        return tensor_data

    def _transform(self,array):
        return (array - array.min())/(array.max()-array.min())
