import sys

import torch
from torch import nn
import numpy as np

from networks.pytorch.nn_lightning import FcNet,LitClusterNet

from tpcutils.data import TPCClusterDataset,SeparatedDataHandler

import glob
import yaml

from matplotlib import pyplot as plt

from config.paths import dpaths as dp
from dotmap import DotMap

def main(select = 'iterationFnet1'):

    config_sel = dp['model_path'] + '/' + select + '/' + 'hyperparams.yml'
    config = DotMap(yaml.safe_load(open(config_sel)))


    files = glob.glob(dp['data_path'] + '/*.txt')
    for f in files:
        print(f.split('/')[-1])

    dataset = TPCClusterDataset(files[0],files[3],transform=config.DATA_PARAMS.NORMALIZE)


    Net = LitClusterNet.load_from_checkpoint(glob.glob(dp['model_path'] + '/' + select + '/' + '*.ckpt')[0])
    Net.eval()


    movData = SeparatedDataHandler(files[2])['xamP']

    target = []
    preds = []
    for i in range(dataset.__len__()):
        tar = dataset.__getitem__(i)

        target.append(tar[1].detach().numpy()[0:7])

        with torch.no_grad():
            yhat = Net(tar[0])

        preds.append(yhat.detach().numpy())

    target = np.array(target)
    preds = np.array(preds)

    f,ax = plt.subplots(2,7,figsize=(22,4))
    #ax = ax.flatten()

    names = ["X",r"$\alpha$","Y","Z",r"$\mathrm{sin}(\phi)$",r"$\mathrm{tan}(\lambda)$",r"$q/p_\mathrm{T}$"]
    for i in range(7):
        ax[0][i].hist2d(target[:,i],preds[:,i],bins=50)
        ax[0][i].set(xlabel='Target', ylabel='Pred')
        ax[0][i].set_title(names[i])

        ax[0][i].axline( (0,0),slope=1,linestyle='--',color='red')

        ax[1][i].hist2d(target[:,i],movData[:,i],bins=50)
        ax[1][i].set(xlabel='Target', ylabel='MovTrackRef')
        ax[1][i].set_title(names[i])

        ax[1][i].axline( (0,0),slope=1,linestyle='--',color='red')

    # ax[-1].set_visible(False)

    plt.tight_layout()

    plt.show()












    return 0


















if __name__=='__main__':

    main()
