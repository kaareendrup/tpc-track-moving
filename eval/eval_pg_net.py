import sys

import torch
from torch import nn
import numpy as np

from networks.pytorch.nn_lightning import PseudoGraphNet
from tpcutils.data import SeparatedDataHandler,read_MC_tracks
from tpcutils.dataset_pt import TPCClusterDataset
from sklearn.model_selection import train_test_split

import glob
import yaml

from matplotlib import pyplot as plt

from config.paths import dpaths as dp
from dotmap import DotMap

import argparse

def main(args):

    Net = PseudoGraphNet.load_from_checkpoint(glob.glob(dp['model_path'] + '/' + args.select + '/' + '*.ckpt')[0])
    Net.eval()

    config = DotMap(yaml.safe_load(open(dp['config1'])))

    iniTrack = config.PATHS.DATA_PATH + '/iniTrack.txt'
    MovTrackRefit = config.PATHS.DATA_PATH + '/movTrackRef.txt'
    MCTrack = config.PATHS.DATA_PATH + '/mcTrack.txt'
    dataset = TPCClusterDataset(iniTrack,MovTrackRefit,transform=config.DATA_PARAMS.NORMALIZE)

    dataset_train,dataset_valid = train_test_split(dataset,test_size=config.DATA_PARAMS.TEST_SIZE, random_state=config.DATA_PARAMS.RANDOM_STATE)

    print("Valid data",len(dataset_valid))

    mcData = dataset._transform_y(read_MC_tracks(MCTrack)[:,2:])

    target = []
    preds = []
    for i in range(dataset.__len__()):
        tar = dataset.__getitem__(i)

        #target.append(tar[1].detach().numpy()[0:7])
        target.append(tar[1].detach().numpy())

        with torch.no_grad():
            yhat = Net(tar[0].unsqueeze(0))

        preds.append(yhat.squeeze(0).detach().numpy())

    target = np.array(target)
    preds = np.array(preds)

    f,ax = plt.subplots(2,5,figsize=(22,4))
    #ax = ax.flatten()

    # names = ["X",r"$\alpha$","Y","Z",r"$\mathrm{sin}(\phi)$",r"$\mathrm{tan}(\lambda)$",r"$q/p_\mathrm{T}$"]
    names = ["Y","Z",r"$\mathrm{sin}(\phi)$",r"$\mathrm{tan}(\lambda)$",r"$q/p_\mathrm{T}$"]
    for i in range(5):
        n_bins = [2000,500] if i==2 else 50

        ax[0][i].hist2d(target[:,i],preds[:,i],bins=n_bins)
        ax[0][i].set(xlabel='Target', ylabel='Pred')
        ax[0][i].set_title(names[i])
        ax[0][i].axline( (0,0),slope=1,linestyle='--',color='red')
    
        ax[1][i].hist2d(target[:,i],mcData[:,i],bins=n_bins)
        ax[1][i].set(xlabel='Target', ylabel='MCTrack')
        ax[1][i].set_title(names[i])
        ax[1][i].axline( (0,0),slope=1,linestyle='--',color='red')

    # ax[-1].set_visible(False)
    ax[0,2].set_xlim(-1,1)
    ax[0,2].set_ylim(-1,1)
    ax[1,2].set_xlim(-1,1)    
    ax[1,2].set_ylim(-1,1)


    plt.tight_layout()

    plt.savefig(dp['model_path'] + '/' + args.select + '/' + 'results/performance.png')
    # plt.show()
    print('Plots saved')

    return 0


if __name__=='__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--select",
                        default="iterationConv2",
                        required=True,
                        help="model directory, in config file known as param MODEL_DIR"
                        )



    args = parser.parse_args()

    main(args)