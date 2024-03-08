import sys

import torch
from torch import nn
import numpy as np

from networks.pytorch.nn_lightning import LitClusterConvolutionalNet, LitClusterNet
from tpcutils.dataset_pt import TPCTreeCluster
from tpcutils.data import SeparatedDataHandler,read_MC_tracks
from sklearn.model_selection import train_test_split

import glob
import yaml

from matplotlib import pyplot as plt

from config.paths import dpaths as dp
from dotmap import DotMap
from scipy.stats import gaussian_kde
import ROOT

import argparse

import mplhep as hep
hep.style.use(hep.style.ALICE)

from array import array
from ROOT import addressof

from tpcio.TreeIO import create_arrays, write_ROOT_TREE

def transform(vec):
    NormArr = np.array([250,250, 1, 5, 40])

    return vec * NormArr

def main(args):

    path_cosmos = '/Users/joachimcarlokristianhansen/st_O2_ML_SC_DS/TPC-analyzer/TPCTracks/models/cosmos'

    pSel = 'FNet_0007_iniRef_Za0_Tgla0_dz_positiveshift'
    TreeName = pSel

    #config_sel = dp['model_path'] + '/' + args.select + '/' + 'logs/version_0/hparams.yaml'
    #config = DotMap(yaml.safe_load(open(config_sel)))
    config = DotMap(yaml.safe_load(open('/Users/joachimcarlokristianhansen/st_O2_ML_SC_DS/TPC-analyzer/TPCTracks/py_dir/config/config_file.yml')))
    configs = DotMap(yaml.safe_load(open(path_cosmos + '/' + f'{pSel}/hyperparams.yml')))


    # Net = LitClusterNet.load_from_checkpoint(glob.glob(dp['model_path'] + '/' + args.select + '/' + '*.ckpt')[0])
    #Net = LitClusterNet.load_from_checkpoint(glob.glob('/Users/joachimcarlokristianhansen/st_O2_ML_SC_DS/TPC-analyzer/TPCTracks/models/aurora/FNet_4_angular/*.ckpt')[0])
    file = glob.glob(path_cosmos + '/' +f'{pSel}/*.ckpt')[-1]
    print("using network file: ",file)
    Net = LitClusterNet.load_from_checkpoint(file)
    Net.eval()
    print("#"*15)
    print("Model successfully loaded...")

    # valid
    file_valid = ROOT.TFile.Open(config.PATHS.DATA_PATH_VALID)
    dataset_valid = TPCTreeCluster(file_valid,transform=True,conf=configs)
    print("Valid data",len(dataset_valid))


    target = []
    preds = []
    preds_normed = []
    data_len = dataset_valid.__len__()
    ini=[]
    clusters = []
    aini_clX, aini_clY, aini_clZ,amov_clX, amov_clY, amov_clZ,aini_clSector, aini_clRow = [],[],[],[],[],[],[],[]
    sel_cluster_idx = []

    mov_vecs = []

    imposedTB,dz = [], []

    for i in range(data_len):
    # for i in range(500):
        sys.stdout.write("\rprocessing %i/%i" % (i+1,data_len))
        sys.stdout.flush()
        #, ini_clY, ini_clZ, mov_clX, mov_clY, mov_clZ,ini_clSector, ini_clRow
        input, tar, ini_vec, mov_vec,ini_clX, ini_clY, ini_clZ,mov_clX, mov_clY, mov_clZ,ini_clSector, ini_clRow, idx_sel  = dataset_valid.__getitem__(i,fVecs=True)
        ini_vec, mov_vec = transform(ini_vec), transform(mov_vec)
        # ini_vecs.append(ini_vec)
        mov_vecs.append(list(mov_vec))

        clusters.append([list(ini_clX),list(ini_clY),list(ini_clZ),list(mov_clX), list(mov_clY), list(mov_clZ),list(ini_clSector),list(ini_clRow)])
        sel_cluster_idx.append(idx_sel)

            

        target.append(tar.detach().numpy())

        ini.append([dataset_valid.tpcIni.iniTrackRef.getY(),dataset_valid.tpcIni.iniTrackRef.getZ(),dataset_valid.tpcIni.iniTrackRef.getSnp(),dataset_valid.tpcIni.iniTrackRef.getTgl(),dataset_valid.tpcIni.iniTrackRef.getQ2Pt()])
        imposedTB.append(dataset_valid.tpcMov.imposedTB)
        dz.append(dataset_valid.tpcMov.dz)

        input = input.unsqueeze(0)

        with torch.no_grad():
            yhat = Net(input)
            yhat = yhat.detach().numpy()
            preds.append(yhat)
            preds_normed.append(transform(yhat))

    print("\n")
    print("Finished predictions loop")
    target = np.array(target)
    # print("target shape:",target.shape)
    preds = np.array(preds).squeeze()
    # print("preds shape:",preds.shape)
    preds_normed = np.array(preds_normed).squeeze()
    # print("preds normed shape:",preds_normed.shape)
    mov_vecs = np.array(mov_vecs)
    sel_cluster_idx = np.array(sel_cluster_idx)
    

    ini = np.array(ini)
    # print("ini shape:", ini.shape)
    imposedTB,dz = np.array(imposedTB), np.array(dz)

    print("Writing ROOT tree")
    write_ROOT_TREE(target,preds,preds_normed,ini,mov_vecs,clusters,sel_cluster_idx,dz,imposedTB,tree_name=f'{TreeName}')
    print("Succesfully completed ROOT tree")

    



    return 0x0



if __name__=='__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--select",
                        default="sliced_TPC_splitted_1",
                        required=False,
                        help="model directory, in config file known as param MODEL_DIR"
                        )



    args = parser.parse_args()

    main(args)
