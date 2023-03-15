import sys

import torch
from torch import nn
import numpy as np

from networks.pytorch.nn_lightning import LitClusterConvolutionalNet
from tpcutils.dataset_pt import TPCClusterDatasetConvolutional
from tpcutils.data import SeparatedDataHandler,read_MC_tracks
from sklearn.model_selection import train_test_split

import glob
import yaml

from matplotlib import pyplot as plt

from config.paths import dpaths as dp
from dotmap import DotMap

import argparse

def main(args):

    config_sel = dp['model_path'] + '/' + args.select + '/' + 'hyperparams.yml'
    config = DotMap(yaml.safe_load(open(config_sel)))


    Net = LitClusterConvolutionalNet.load_from_checkpoint(glob.glob(dp['model_path'] + '/' + args.select + '/' + '*.ckpt')[0])
    Net.eval()

    print("#"*15)
    print("Model successfully loaded...")

    iniTrack = config.PATHS.DATA_PATH + '/iniTrack.npy'
    MovTrackRefit = config.PATHS.DATA_PATH + '/movTrackRef.npy'

    dataset = TPCClusterDatasetConvolutional(iniTrack,MovTrackRefit,
                                            transform=config.DATA_PARAMS.NORMALIZE,
                                            nTPCclusters=config.DATA_PARAMS.TPC_CLUSTERS,
                                            np_data=config.DATA_PARAMS.NUMPY_DATA)
    # legacy data
    # files = glob.glob(dp['data_path'] + '/*.txt')
    # for f in files:
    #     print(f.split('/')[-1])
    #
    # dataset = TPCClusterDatasetConvolutional(files[0],files[2],
    #                                         transform=config.DATA_PARAMS.NORMALIZE,
    #                                         nTPCclusters=config.DATA_PARAMS.TPC_CLUSTERS)

    dataset_train,dataset_valid = train_test_split(dataset,test_size=config.DATA_PARAMS.TEST_SIZE, random_state=config.DATA_PARAMS.RANDOM_STATE)

    print("Valid data",len(dataset_valid))

    #mcData = read_MC_tracks(files[3])[:,2:]
    mcData_file = config.PATHS.DATA_PATH + '/mcTrack.npy'
    mcData = read_MC_tracks(mcData_file)[:,2:]
    print(mcData.shape)

    total_target = []
    for i in range(dataset.__len__()):
        tar = dataset.__getitem__(i)
        total_target.append(tar['target'].detach().numpy())
    total_target = np.array(total_target)

    print("Monte carlo data shape: {}".format(mcData.shape))
    print("Total target data shape: {}".format(total_target.shape))

    target = []
    preds = []
    for i in range(dataset_valid.__len__()):
        tar = dataset_valid.__getitem__(i)
        xyz = tar['input_xyz_row']
        mP = tar['mP']
        target.append(tar['target'].detach().numpy())

        with torch.no_grad():
            yhat = Net(xyz,mP)


        preds.append(yhat.detach().numpy())

    target = np.array(target)
    preds = np.array(preds)

    print("Valid target data shape: {}".format(target.shape))
    print("Prediction valid data shape: {}".format(preds.shape))

    f,ax = plt.subplots(2,5,figsize=(16,4))
    #ax = ax.flatten()

    names = ["Y","Z",r"$\mathrm{sin}(\phi)$",r"$\mathrm{tan}(\lambda)$",r"$q/p_\mathrm{T}$"]
    for i in range(5):
        ax[0][i].hist2d(preds[:,i],target[:,i],bins=50)
        ax[0][i].set(ylabel='MovTrackRefit', xlabel='NN Prediction')
        ax[0][i].set_title("{}".format(names[i]))

        ax[0][i].axline( (0,0),slope=1,linestyle='--',color='red',linewidth = 0.5)

        ax[1][i].hist2d(mcData[:,i],total_target[:,i],bins=50)
        ax[1][i].set(ylabel='IniTrackRefit', xlabel='MCTrack')
        ax[1][i].set_title("{}".format(names[i]))

        ax[1][i].axline( (0,0),slope=1,linestyle='--',color='red',linewidth = 0.5)

    # ax[-1].set_visible(False)

    plt.tight_layout()

    plt.show()

    # f.savefig("pred.png",bbox_inches='tight')




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
