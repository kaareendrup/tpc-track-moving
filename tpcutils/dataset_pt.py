import sys,os

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
import operator

import ROOT

from tpcutils.data import DataHandler,SeparatedDataHandler
from tpcutils.data import select_tpc_clusters_idx

# ROOT.gInterpreter.ProcessLine('#include "../tpcio/TrackTPC.h"')
ROOT.gInterpreter.ProcessLine('#include "/Users/joachimcarlokristianhansen/st_O2_ML_SC_DS/TPC-analyzer/TPCTracks/py_dir/tpcio/TrackTPC.h"')
#### PYTORCH

#Legacy
class TPCClusterDataset(Dataset):
    def __init__(self, tracks_path, mov_path, transform=False,np_data=True):

        self.X = DataHandler(tracks_path,np_data)
        self.y = SeparatedDataHandler(mov_path,np_data)['xamP'][:,2:]


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

#Legacy
class TPCClusterDatasetConvolutional(Dataset):
    def __init__(self, tracks_path, mov_path,TPC_settings, transform=False,tpcNorm=False,np_data=True):



        self.X = SeparatedDataHandler(tracks_path,TPC_settings,np_data)
        self.Y = SeparatedDataHandler(mov_path,TPC_settings,np_data)
        self.y = self.Y['xamP']

        if tpcNorm:
            self.X['xyz'] = (self.X['xyz'] + 260)/(260+260)


        self._TPCDistortions = self.X['xyz'] - self.Y['xyz']
        self._PADDists = self.X['pads']

        self.x = np.column_stack((self._TPCDistortions,self._PADDists[:,np.newaxis,:]))

        self.transform = transform

        self.index_update = 6

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


class TPCTreeCluster(Dataset):
    def __init__(self, file,transform=False,conf=None):

        self._file = file

        self.tpcIni = self._file.Get("tpcIni")
        self.tpcMov = self._file.Get("tpcMov")
        # tree_key tpcIni has key: iniTrackRef
        # tree_key tpcMov has key: movTrackRef

        self.EntriesIni = self.tpcIni.GetEntries()
        self.EntriesMov = self.tpcMov.GetEntries() 


        self.tpcMaxRow = 159 # 159 rows/max number of tpc clusters for padding

        self.nKalmanFits = 6

        self.transform=transform


        if conf is not None:
            self.config= conf

        clusters = self.config.DATA_PARAMS.TPC_SETTINGS.TPC_CLUSTERS
        self.__shape = 7 + (clusters*5) # 7 ini params, clx,cly,clz,sector,row



    def _padForClusters(self,array):
        return np.pad(array,(0,self.tpcMaxRow-len(array)),"constant")

    def _shape(self):
        return self.__shape


    def __iniConstruct(self,):

        #construct ini vector
        ini_clSector = np.array(self.tpcIni.clSector)
        ini_clRow = np.array(self.tpcIni.clRow)

        iniX = self.tpcIni.iniTrackRef.getX()

        iniAlpha = self.tpcIni.iniTrackRef.getAlpha()

        iniY = self.tpcIni.iniTrackRef.getY()
        iniZ = self.tpcIni.iniTrackRef.getZ()
        iniSnp = self.tpcIni.iniTrackRef.getSnp()
        iniTgl = self.tpcIni.iniTrackRef.getTgl()
        iniQ2Pt = self.tpcIni.iniTrackRef.getQ2Pt()

        #ini toc clusters
        ini_clX = self.tpcIni.clX
        ini_clY = self.tpcIni.clY
        ini_clZ = self.tpcIni.clZ

        #ini_counter = self.tpcIni.counter

        ini_vec1 = np.array([iniX, iniAlpha])
        ini_vec2 = np.array([iniY, iniZ, iniSnp, iniTgl, iniQ2Pt])
        
        return ini_vec1,ini_vec2, ini_clX, ini_clY, ini_clZ, ini_clSector, ini_clRow

    def __movConstruct(self):

        #construct mov vector or rather target
        MovY = self.tpcMov.movTrackRef.getY()
        MovZ = self.tpcMov.movTrackRef.getZ()
        MovSnp = self.tpcMov.movTrackRef.getSnp()
        MovTgl = self.tpcMov.movTrackRef.getTgl()
        MovQ2Pt = self.tpcMov.movTrackRef.getQ2Pt()

        np_target = np.array([ MovY, MovZ, MovSnp, MovTgl, MovQ2Pt ])

        #construct moved tpc clusters
        #mov_clX = np.array(self.tpcMov.clX)
        #mov_clY = np.array(self.tpcMov.clY)
        #mov_clZ = np.array(self.tpcMov.clZ)
        mov_clX = self.tpcMov.clX
        mov_clY = self.tpcMov.clY
        mov_clZ = self.tpcMov.clZ

        # n_copy = self.tpcMov.copy
        # maxCopy = self.tpcMov.maxCopy

        # mov_counter = self.tpcMov.counter

        return np_target, mov_clX, mov_clY, mov_clZ#, mov_counter, n_copy, maxCopy

    def __match_tracks(self):

        counter = self.tpcMov.counter
        # n_copy = self.tpcMov.copy
        # maxCopy = self.tpcMov.maxCopy

        self.tpcIni.GetEntry(counter)



    def _getDistortionEffects(self,arr1,arr2):
        diff = list(map(operator.sub, arr1, arr2))
        return diff

    def _checkVectorlen_(self,array):

        if len(array) != self.__shape:
            return np.pad(array,(0,self.__shape-len(array)),"constant")
        else:
            return array


    def __getitem__(self,idx):

        self.tpcMov.GetEntry(idx)   # we follow mov indexing and match the counter to ini
        self.__match_tracks()


        #print("mov count",self.tpcMov.counter)
        #print("ini count",self.tpcIni.counter)
        if self.tpcMov.counter!=self.tpcIni.counter:
            print("Mov",self.tpcMov.counter)
            print("Ini",self.tpcIni.counter)
            print("Tracks are matched incorrectly, exiting...")
            sys.exit(1)

        np_target, mov_clX, mov_clY, mov_clZ = self.__movConstruct()

        ini_vec1,ini_vec2, ini_clX, ini_clY, ini_clZ, ini_clSector, ini_clRow = self.__iniConstruct()



        xDist = np.array(self._getDistortionEffects(mov_clX,ini_clX))
        yDist = np.array(self._getDistortionEffects(mov_clY,ini_clY))
        zDist = np.array(self._getDistortionEffects(mov_clZ,ini_clZ))
        
        if not self.transform:
            ini_clSector = self._padForClusters(ini_clSector)
            ini_clRow = self._padForClusters(ini_clRow)

            xDist = self._padForClusters(xDist)
            yDist = self._padForClusters(yDist)
            zDist = self._padForClusters(zDist)
        else:
            if self.config is not None:
                idx_sel = select_tpc_clusters_idx(self.config.DATA_PARAMS.TPC_SETTINGS,len(xDist)-1)
            else:
                idx_sel = select_tpc_clusters_idx(config.DATA_PARAMS.TPC_SETTINGS,len(xDist)-1)

            ini_clSector = ini_clSector[idx_sel]
            ini_clRow = ini_clRow[idx_sel]
            xDist = xDist[idx_sel]
            yDist = yDist[idx_sel]
            zDist = zDist[idx_sel]


        #concatenating everything (easier to implement in O2)
        input_vector = np.concatenate((ini_vec1,ini_vec2,xDist,yDist,zDist,ini_clSector, ini_clRow))
        #checking input vector and padding with zeros if it doesn't match the length
        input_vector = self._checkVectorlen_(input_vector)

        #convert to tensor
        input_pt = torch.from_numpy(input_vector).float()
        target_pt = torch.from_numpy(np_target).float()


        return input_pt, target_pt
    

    def __len__(self):
        return self.EntriesMov
