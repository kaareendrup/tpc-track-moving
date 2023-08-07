import sys

import torch
from torch import nn
import numpy as np

from networks.pytorch.nn_lightning import PseudoGraphNetSingle
from tpcutils.dataset_pt import TPCTreeCluster
from tpcutils.data import SeparatedDataHandler, read_MC_tracks
from sklearn.model_selection import train_test_split

import glob
import yaml
from tqdm import tqdm

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

from config.paths import dpaths as dp
from dotmap import DotMap
from scipy.stats import gaussian_kde
import ROOT

import argparse

# import mplhep as hep
# hep.style.use(hep.style.ALICE)


def main(which, args):


    config = DotMap(yaml.safe_load(open('/home/kaare/alice/tpc-track-moving/config/config_file_root.yml')))

    # Net = PseudoGraphNet.load_from_checkpoint('/home/kaare/alice/ServiceTask/models/pytorch/PseudoGraph01/PseudoGraph_epoch=1-val_loss=0.10.ckpt')
    Net = PseudoGraphNetSingle.load_from_checkpoint('/home/kaare/alice/ServiceTask/models/pytorch/PseudoGraph01/PseudoGraph_epoch=11-val_loss=0.05.ckpt')
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

    for i in tqdm(range(data_len)):
    # for i in tqdm(range(100)):
        # sys.stdout.write("\rprocessing %i/%i" % (i+1,data_len))
        # sys.stdout.flush()

        input, tar = dataset_valid.__getitem__(i)

        ini.append([dataset_valid.tpcIni.iniTrackRef.getY(),dataset_valid.tpcIni.iniTrackRef.getZ(),dataset_valid.tpcIni.iniTrackRef.getSnp(),dataset_valid.tpcIni.iniTrackRef.getTgl(),dataset_valid.tpcIni.iniTrackRef.getQ2Pt()])
        imposedTB.append(dataset_valid.tpcMov.imposedTB)
        dz.append(dataset_valid.tpcMov.dz)

        target.append(tar.detach().numpy())
        # input = input.unsqueeze(0).unsqueeze(0)
        input = input.unsqueeze(0)

        with torch.no_grad():
            yhat = Net(input)
            
            preds.append(yhat.detach().numpy())

    target = np.array(target)
    preds = np.array(preds).squeeze()

    ini = np.array(ini)
    imposedTB,dz = np.array(imposedTB), np.array(dz)
    

    # write_ROOT_TREE(target,preds,ini,dz,imposedTB,tree_name='RNN')
    # print("Finished writing tree")
    # print("Valid target data shape: {}".format(target.shape))
    # print("Prediction valid data shape: {}".format(preds.shape))

    f,ax = plt.subplots(1,1,figsize=(10,8))
    #ax = ax.flatten()

    names = ["Y","Z",r"$\mathrm{sin}(\phi)$",r"$\lambda$",r"$q/p_\mathrm{T}$"]
    # lims = np.array([[-25,25],[-200,200],[-np.pi,np.pi],[-2.4,2.4],[-25,25]])
    lims = np.array([[-25,25],[-20,20],[-1,1],[-2.4,2.4],[-25,25]])
    bins = np.array([[50,50], [500,50], [50,50], [50,50], [50,50]])
    # skal plotte prediction:
    # MoveTrackRefit" vs "iniTrackRef - MoveTrackRefit" (for Z should be iniTrackRef.getZ() - dz - MoveTrackRefit.getZ()
    #

    text_size = 10
    # y = preds[:,i]
    # x = target[:,i]
    # xy = np.vstack([x,y])
    # z = gaussian_kde(xy)(xy)
    # idx = z.argsort()
    # x, y, z = x[idx], y[idx], z[idx]
    # ax[i].scatter(x, y, c=z, s=10)

    ax.hist2d(target[:,which],preds[:,0],bins=bins[which])
    ax.set_ylabel('NN Prediction',size=text_size)
    ax.set_xlabel('MovTrackRefit',size=text_size)
    ax.set_title("{}".format(names[which]),size=text_size)

    ax.set_xlim(*lims[which])
    ax.set_ylim(*lims[which])

    ax.tick_params(axis='both', which='major', labelsize=text_size)

    ax.set_aspect('equal')

    ax.axline( (0,0),slope=1,linestyle='--',color='red',linewidth = 0.5)

    # ax[-1].set_visible(False)

    plt.tight_layout()
    plt.show()
    f.savefig("PGNetpred.png",bbox_inches='tight')

    return 0



if __name__=='__main__':

    which = 3
    parser = argparse.ArgumentParser()

    parser.add_argument("--select",
                        default="sliced_TPC_splitted_1",
                        required=False,
                        help="model directory, in config file known as param MODEL_DIR"
                        )



    args = parser.parse_args()

    main(which, args)
