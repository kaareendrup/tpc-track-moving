import sys,os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


import copy

def GetData(path, test_size=0.25, random_state=42):

    df = pd.read_csv(path,index_col=0)

    train,test = train_test_split(df, test_size=test_size, random_state=random_state)

    X = train.values[:,0:5]
    y = train.values[:,5:]
    # X = X[:,np.newaxis,:]
    # y = y[:,np.newaxis,:]


    X_test = test.values[:,0:5]
    y_test = test.values[:,5:]
    # X_test = X_test[:,np.newaxis,:]
    # y_test = y_test[:,np.newaxis,:]

    X_scaled = copy.deepcopy(X)
    y_scaled = copy.deepcopy(y)
    X_test_scaled = copy.deepcopy(X_test)
    y_test_scaled = copy.deepcopy(y_test)
    for row in range(len(X)):
        X_scaled[row,:] = (X[row,:] - X[row,:].min())/(X[row,:].max()-X[row,:].min())

        y_scaled[row,:] = (y[row,:] - y[row,:].min())/(y[row,:].max()-y[row,:].min())


    for row in range(len(X_test)):
        X_test_scaled[row,:] = (X_test[row,:] - X_test[row,:].min())/(X_test[row,:].max()-X_test[row,:].min())

        y_test_scaled[row,:] = (y_test[row,:] - y_test[row,:].min())/(y_test[row,:].max()-y_test[row,:].min())


    return df, train, test, X, y, X_test, y_test, X_scaled, y_scaled, X_test_scaled, y_test_scaled


def read_nonMC_tracks(data_path,np_data=True):
    data_names = ["X","alpha","Y","Z","sin_phi","tgLambda","q2pt","bcTB","dz","cov1","cov2","cov3","cov4","cov5","cov6","cov7","cov8",
                 "cov9","cov10","cov11","cov12","cov13","cov14","cov15"]
    nClusters=159

    if np_data:
        Track = np.load(data_path)

        cluster_xyz_data=Track[:,len(data_names)+nClusters*2:]

        sector_data = Track[:,len(data_names):len(data_names)+nClusters]

        row_data = Track[:,len(data_names)+nClusters:len(data_names)+nClusters*2]

        mP_vector_data = Track[:,0:len(data_names)-(15+2)]

    else:
        Track = pd.read_csv(data_path,header=None,sep=' ',index_col=0)#names=data_names)

        cluster_xyz_data=Track.iloc[:,len(data_names)+nClusters*2:-1]

        sector_data = Track.iloc[:,len(data_names):len(data_names)+nClusters]

        row_data = Track.iloc[:,len(data_names)+nClusters:len(data_names)+nClusters*2]

        mP_vector_data = Track.iloc[:,0:len(data_names)-(15+2)]

    return cluster_xyz_data, sector_data, row_data, mP_vector_data


def get_clusters_xyz_lab_coord(xyz_data,sector_data,iTrack,np_data=True):

    if not np_data:
        xyz = xyz_data.iloc[[iTrack]].to_numpy()
        #print(iTrack)
    else:
        xyz = xyz_data[iTrack]

    xyz = np.reshape(xyz, (3,-1), order='F')


    cut = np.where(xyz==0)[1][0]

    xyz = xyz[:,:cut]

    # Correct for sector
    if not np_data:
        sector = sector_data.iloc[[iTrack]].to_numpy()
    else:
        sector = sector_data[iTrack]


    if not np_data:
        sector = sector[0,:cut]
    else:
        sector = sector[:cut]
    sector_corr = - sector * 20/360*2*np.pi

    x_new = xyz[0] * np.cos(sector_corr) + xyz[1] * np.sin(sector_corr)
    y_new = - xyz[0] * np.sin(sector_corr) + xyz[1] * np.cos(sector_corr)

    return x_new,y_new, xyz[2]

def GetClusterData(data,i=0,nTPCclusters=20,np_data=True):

    clusters_xyz, sectors, pad_rows, XmP = data


    x_new,y_new,z_new = get_clusters_xyz_lab_coord(clusters_xyz,sectors,i,np_data)

    idx = np.round(np.linspace(0, len(x_new) - 1, nTPCclusters)).astype(int)

    if np_data:
        pads = pad_rows[i]

        return XmP[i], x_new[idx], y_new[idx], z_new[idx], pads[idx]
    else:

        pads = pad_rows.iloc[[i]].to_numpy().squeeze()


        return XmP.iloc[[i]].to_numpy().squeeze(), x_new[idx], y_new[idx], z_new[idx], pads[idx]

def read_MC_tracks(data_path):

    data_names = ["X","alpha","Y","Z","sin_phi","tgLambda","q2pt","bcTB","dz","cov1","cov2","cov3","cov4","cov5","cov6","cov7","cov8",
                 "cov9","cov10","cov11","cov12","cov13","cov14","cov15"]

    tracks = pd.read_csv(data_path,header=None,sep=' ',index_col=0)#names=data_names)
    tracks = tracks.to_numpy()[:,:-1]

    tracks = tracks[:,0:len(data_names)-(15+2)]


    return tracks

def DataHandler(path,nTPCclusters=20,np_data=True):

    data = read_nonMC_tracks(path,np_data)


    iTracks = data[0].shape[0]

    X = []
    for track in range(iTracks):
        temp = GetClusterData(data,track,nTPCclusters)

        temp2 = np.concatenate([*temp])

        X.append(temp2)

    return np.array(X)

def SeparatedDataHandler(path,nTPCclusters=20,np_data=True):
    """
        Data handler for generating xmp vector, 3xN cluster vector plus 1xN row vector
    """
    data = read_nonMC_tracks(path,np_data)

    iTracks = data[0].shape[0]
    fXAmP = []
    fx = []
    fy = []
    fz = []
    fpads = []

    for track in range(iTracks):
        XAmP, x,y,z, pad_rows = GetClusterData(data,track,nTPCclusters,np_data)

        fXAmP.append(XAmP)
        fx.append(x)
        fy.append(y)
        fz.append(z)
        fpads.append(pad_rows)

    fx = np.array(fx)
    fy = np.array(fy)
    fz = np.array(fz)


    data_dict = {}
    data_dict['xamP'] = np.array(fXAmP)
    data_dict['xyz'] = np.array([fx,fy,fz]).transpose(1,0,2) # pd 1 0 2
    data_dict['pads'] = np.array(fpads)

    return data_dict


def getAllData(tracks_path,mov_path, test_size=0.25, random_state=42):

    X = DataHandler(tracks_path)
    y = SeparatedDataHandler(mov_path)['xamP'][:,2:]

    X_scaled = copy.deepcopy(X)
    y_scaled = copy.deepcopy(y)

    for row in range(len(X)):
        X_scaled[row,:] = (X[row,:] - X[row,:].min())/(X[row,:].max()-X[row,:].min())
        y_scaled[row,:] = (y[row,:] - y[row,:].min())/(y[row,:].max()-y[row,:].min())


    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test





if __name__=='__main__':















    print("done")
