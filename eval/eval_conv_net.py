import sys

import torch
from torch import nn
import numpy as np

from networks.pytorch.nn_lightning import LitClusterConvolutionalNet
from tpcutils.data import TPCClusterDatasetConvolutional,SeparatedDataHandler,read_MC_tracks

import glob
import yaml

from matplotlib import pyplot as plt

from config.paths import dpaths as dp
from dotmap import DotMap

def main(select = 'iterationConv1'):

    config_sel = dp['model_path'] + '/' + select + '/' + 'hyperparams.yml'
    config = DotMap(yaml.safe_load(open(config_sel)))


    Net = LitClusterConvolutionalNet.load_from_checkpoint(glob.glob(dp['model_path'] + '/' + select + '/' + '*.ckpt')[0])
    Net.eval()



    files = glob.glob(dp['data_path'] + '/*.txt')
    for f in files:
        print(f.split('/')[-1])

    dataset = TPCClusterDatasetConvolutional(files[0],files[2],
                                            transform=config.DATA_PARAMS.NORMALIZE,
                                            nTPCclusters=config.DATA_PARAMS.TPC_CLUSTERS)

    movData = read_MC_tracks(files[3])[:,2:]


    target = []
    preds = []
    for i in range(dataset.__len__()):
        tar = dataset.__getitem__(i)
        xyz = tar['input_xyz_row']
        mP = tar['mP']
        target.append(tar['target'].detach().numpy())

        with torch.no_grad():
            yhat = Net(xyz,mP)


        preds.append(yhat.detach().numpy())

    target = np.array(target)
    preds = np.array(preds)

    f,ax = plt.subplots(2,5,figsize=(22,4))
    #ax = ax.flatten()

    names = ["Y","Z",r"$\mathrm{sin}(\phi)$",r"$\mathrm{tan}(\lambda)$",r"$q/p_\mathrm{T}$"]
    for i in range(5):
        ax[0][i].hist2d(target[:,i],preds[:,i],bins=50)
        ax[0][i].set(xlabel='Target', ylabel='Pred')
        ax[0][i].set_title(names[i])

        ax[0][i].axline( (0,0),slope=1,linestyle='--',color='red',linewidth = 0.5)

        ax[1][i].hist2d(target[:,i],movData[:,i],bins=50)
        ax[1][i].set(xlabel='Target', ylabel='MovTrackRef')
        ax[1][i].set_title(names[i])

        ax[1][i].axline( (0,0),slope=1,linestyle='--',color='red',linewidth = 0.5)

    # ax[-1].set_visible(False)

    plt.tight_layout()

    plt.show()

    f.savefig("pred.png",bbox_inches='tight')












    return 0


















if __name__=='__main__':

    main()
