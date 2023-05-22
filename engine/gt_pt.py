import sys

from tpcutils.dataset_pt import TPCTreeCluster

from networks.pytorch.nn_lightning import FcNet,LitClusterNet
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



def generalised_trainer_PT_clusters(**kwargs):

    config = DotMap(yaml.safe_load(open('/Users/joachimcarlokristianhansen/st_O2_ML_SC_DS/TPC-analyzer/TPCTracks/py_dir/config/config_file.yml')))

    if config.DATA_PARAMS.IS_ROOT:
        print("Using the tpc-trackStudy file in ROOT format")
        # train
        file = ROOT.TFile.Open(config.PATHS.DATA_PATH_TRAIN)
        dataset_train = TPCTreeCluster(file,transform=True,conf=config)
        # valid
        file_valid = ROOT.TFile.Open(config.PATHS.DATA_PATH_VALID)
        dataset_valid = TPCTreeCluster(file_valid,transform=True,conf=config)




    #dataset_train,dataset_valid = train_test_split(dataset,test_size=config.DATA_PARAMS.TEST_SIZE, random_state=config.DATA_PARAMS.RANDOM_STATE)

    print("Initializing data loaders")
    train_loader = DataLoader(dataset_train,
                              batch_size=config.HYPER_PARAMS.BATCH_SIZE,
                              shuffle=config.DATA_PARAMS.SHUFFLE_TRAIN,
                              num_workers=config.DATA_PARAMS.NUM_WORKERS)
    val_loader = DataLoader(dataset_valid,
                            batch_size=config.HYPER_PARAMS.BATCH_SIZE,
                            shuffle=config.DATA_PARAMS.SHUFFLE_VALID,
                            num_workers=config.DATA_PARAMS.NUM_WORKERS,
                            )

    print("Initializing net")
    #input shape: 7+nClustersSelected*3
    model = LitClusterNet(32,config)

    logger = TensorBoardLogger(name="logs",save_dir=config.PATHS.SAVE_PATH + '/' + config.PATHS.MODEL_DIR)
    # training
    trainer = pl.Trainer(
                    num_nodes=config.PYTORCH_LIGHTNING_PARAMS.NUM_NODES,
                    precision=config.PYTORCH_LIGHTNING_PARAMS.PRECISION,
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
                               StochasticWeightAveraging(swa_lrs=config.HYPER_PARAMS.SWA_LRS),
                               ],
                    max_epochs=config.HYPER_PARAMS.MAX_EPOCHS,
                    logger=logger
                    )

    print("Attempting to fit")
    trainer.fit(model, train_loader, val_loader,)

    print("Fit finished successfully, dumping config to json...")
    with io.open(config.PATHS.SAVE_PATH + '/' + config.PATHS.MODEL_DIR + '/hyperparams.yml', 'w', encoding='utf8') as outfile:
        yaml.dump(config.toDict(),outfile)






if __name__=='__main__':

    generalised_trainer_PT_clusters()
