import sys

from tpcutils.dataset_pt import TPCClusterDataset

from networks.pytorch.nn_lightning import PseudoGraphNet
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc
import glob
from sklearn.model_selection import train_test_split

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.loggers import TensorBoardLogger

import ROOT
import yaml
import io

from dotmap import DotMap

from config.paths import dpaths as dp

def generalised_trainer_pseudo_graph(**kwargs):


    # config = Struct(**yaml.safe_load(open(dp['config1'])))
    config = DotMap(yaml.safe_load(open(dp['config1'])))

    #files = glob.glob(config.PATHS.DATA_PATH + '/*.txt')
    #dataset = TPCClusterDataset(files[0],files[2],transform=config.DATA_PARAMS.NORMALIZE)

    if config.DATA_PARAMS.ROOT:
        print("Using the tpc-trackStudy file in ROOT format")
        file = ROOT.TFile.Open(config.PATHS.DATA_PATH)
        dataset = TPCTreeCluster(file,transform=True,conf=config)

    if config.DATA_PARAMS.NUMPY_DATA:
        print("Using NUMPY data")
        iniTrack = config.PATHS.DATA_PATH + '/iniTrack.npy'
        MovTrackRefit = config.PATHS.DATA_PATH + '/movTrackRef.npy'

        dataset = TPCClusterDatasetConvolutional(iniTrack,MovTrackRefit,
                                                transform=config.DATA_PARAMS.NORMALIZE,
                                                TPC_settings=config.DATA_PARAMS.TPC_SETTINGS,
                                                np_data=config.DATA_PARAMS.NUMPY_DATA)
    else:
        print("Using txt data")
        files = glob.glob(config.PATHS.DATA_PATH + '/*.txt')

        dataset = TPCClusterDatasetConvolutional(files[0],files[2],
                                                transform=config.DATA_PARAMS.NORMALIZE,
                                                TPC_settings=config.DATA_PARAMS.TPC_SETTINGS,
                                                np_data=config.DATA_PARAMS.NUMPY_DATA)


    dataset_train,dataset_valid = train_test_split(dataset,test_size=config.DATA_PARAMS.TEST_SIZE, random_state=config.DATA_PARAMS.RANDOM_STATE)

    train_loader = DataLoader(dataset_train,
                              batch_size=config.HYPER_PARAMS.BATCH_SIZE,
                              shuffle=config.DATA_PARAMS.SHUFFLE_TRAIN,
                              num_workers=config.DATA_PARAMS.NUM_WORKERS)
    val_loader = DataLoader(dataset_valid,
                            batch_size=config.HYPER_PARAMS.BATCH_SIZE,
                            shuffle=config.DATA_PARAMS.SHUFFLE_VALID,
                            num_workers=config.DATA_PARAMS.NUM_WORKERS)

    model = PseudoGraphNet(dataset._shape(), config, train_loader)

    logger = TensorBoardLogger(name="logs",save_dir=config.PATHS.SAVE_PATH + '/' + config.PATHS.MODEL_DIR)

    # training
    trainer = pl.Trainer(
                    num_nodes=config.PYTORCH_LIGHTNING_PARAMS.NUM_NODES,
                    # precision=config.PYTORCH_LIGHTNING_PARAMS.PRECISION,
                    limit_train_batches=config.PYTORCH_LIGHTNING_PARAMS.LIMIT_TRAIN_BATCHES,
                    accelerator=config.PYTORCH_LIGHTNING_PARAMS.ACCELERATOR,
                    devices=config.PYTORCH_LIGHTNING_PARAMS.DEVICES,
                    callbacks=[EarlyStopping(monitor="val_loss",
                                             mode="min",
                                             patience=config.HYPER_PARAMS.EARLY_STOPPING.PATIENCE,
                                             verbose=0),
                               ModelCheckpoint(dirpath = config.PATHS.SAVE_PATH + '/' + config.PATHS.MODEL_DIR,
                                               filename =config.MODEL.NAME + '_' + '{epoch}-{val_loss:.2f}',
                                               monitor="val_loss",
                                               mode='min',
                                               save_top_k=1),
                            #    StochasticWeightAveraging(swa_lrs=config.HYPER_PARAMS.SWA_LRS),
                               ],
                    max_epochs=config.HYPER_PARAMS.MAX_EPOCHS,
                    logger=logger,
                    log_every_n_steps=config.PYTORCH_LIGHTNING_PARAMS.LOG_EVERY_N_STEPS
                    )


    trainer.fit(model, train_loader, val_loader,)



    with io.open(config.PATHS.SAVE_PATH + '/' + config.PATHS.MODEL_DIR + '/hyperparams.yml', 'w', encoding='utf8') as outfile:
        yaml.dump(config.toDict(),outfile)






if __name__=='__main__':

    generalised_trainer_pseudo_graph()
