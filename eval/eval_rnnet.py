import sys

import torch
from torch import nn
import numpy as np

from networks.pytorch.nn_lightning import LitRNN
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

def main(args):

    #config_sel = dp['model_path'] + '/' + args.select + '/' + 'logs/version_0/hparams.yaml'
    #config = DotMap(yaml.safe_load(open(config_sel)))
    config = DotMap(yaml.safe_load(open('/Users/joachimcarlokristianhansen/st_O2_ML_SC_DS/TPC-analyzer/TPCTracks/py_dir/config/config_file.yml')))



    # Net = LitClusterNet.load_from_checkpoint(glob.glob(dp['model_path'] + '/' + args.select + '/' + '*.ckpt')[0])
    Net = LitRNN.load_from_checkpoint('/Users/joachimcarlokristianhansen/st_O2_ML_SC_DS/TPC-analyzer/TPCTracks/models/aurora/RNN_4/RNNet_epoch=13-val_loss=18.10.ckpt')
    Net.eval()
    print("#"*15)
    print("Model successfully loaded...")

    # valid
    file_valid = ROOT.TFile.Open(config.PATHS.DATA_PATH_VALID)
    dataset_valid = TPCTreeCluster(file_valid,transform=True,conf=config)
    print("Valid data",len(dataset_valid))


    target = []
    preds = []
    data_len = dataset_valid.__len__()
    ini=[]

    imposedTB,dz = [], []

    for i in range(data_len):
        sys.stdout.write("\rprocessing %i/%i" % (i+1,data_len))
        sys.stdout.flush()

        input, tar = dataset_valid.__getitem__(i)

        ini.append([dataset_valid.tpcIni.iniTrackRef.getY(),dataset_valid.tpcIni.iniTrackRef.getZ(),dataset_valid.tpcIni.iniTrackRef.getSnp(),dataset_valid.tpcIni.iniTrackRef.getTgl(),dataset_valid.tpcIni.iniTrackRef.getQ2Pt()])
        imposedTB.append(dataset_valid.tpcMov.imposedTB)
        dz.append(dataset_valid.tpcMov.dz)

        target.append(tar.detach().numpy())
        input = input.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            yhat,_ = Net(input)
            
            preds.append(yhat.detach().numpy())

    target = np.array(target)
    preds = np.array(preds).squeeze()

    ini = np.array(ini)
    imposedTB,dz = np.array(imposedTB), np.array(dz)
    

    write_ROOT_TREE(target,preds,ini,dz,imposedTB,tree_name='RNN')
    print("Finished writing tree")
    print("Valid target data shape: {}".format(target.shape))
    print("Prediction valid data shape: {}".format(preds.shape))

    f,ax = plt.subplots(1,5,figsize=(16,4))
    #ax = ax.flatten()

    names = ["Y","Z",r"$\mathrm{sin}(\phi)$",r"$\lambda$",r"$q/p_\mathrm{T}$"]
    lims = np.array([[-25,25],[-200,200],[-np.pi,np.pi],[-2.4,2.4],[-25,25]])
    # skal plotte prediction:
    # MoveTrackRefit" vs "iniTrackRef - MoveTrackRefit" (for Z should be iniTrackRef.getZ() - dz - MoveTrackRefit.getZ()
    #

    text_size = 10
    for i in range(5):
        y = preds[:,i]
        x = target[:,i]
        xy = np.vstack([x,y])
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
        ax[i].scatter(x, y, c=z, s=10)
        #ax[i].hist2d(preds[:,i],target[:,i],bins=50)



        ax[i].set_xlabel('MovTrackRefit',size=text_size)
        ax[i].set_ylabel('NN Prediction',size=text_size)
        ax[i].set_title("{}".format(names[i]),size=text_size)

        ax[i].set_xlim(*lims[i])
        ax[i].set_ylim(*lims[i])

        ax[i].tick_params(axis='both', which='major', labelsize=text_size)

        ax[i].set_aspect('equal')

        ax[i].axline( (0,0),slope=1,linestyle='--',color='red',linewidth = 0.5)

    # ax[-1].set_visible(False)

    plt.tight_layout()

    plt.show()

    f.savefig("RNNpred.png",bbox_inches='tight')




    return 0



if __name__=='__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--select",
                        default="sliced_TPC_splitted_1",
                        required=False,
                        help="model directory, in config file known as param MODEL_DIR"
                        )



    args = parser.parse_args()

    main(args)
