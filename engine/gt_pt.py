import sys

sys.path.append("/Users/joachimcarlokristianhansen/st_O2_ML_SC_DS/TPC-analyzer/TPCTracks/py_dir")


from tpcutils.data import TPCClusterDataset
from tpcutils.dot_not import AttributeDict,Struct

from networks.pytorch.nn_pt import FcNet,LitClusterNet
#from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
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

import yaml

def generalised_trainer_TF_clusters(**kwargs):


    config = Struct(**yaml.safe_load(open('/Users/joachimcarlokristianhansen/st_O2_ML_SC_DS/TPC-analyzer/TPCTracks/py_dir/config/config_file.yml')))


    files = glob.glob(config.PATHS.DATA_PATH + '/*.txt')
    dataset = TPCClusterDataset(files[0],files[3],transform=config.DATA_PARAMS.NORMALIZE)

    dataset_train,dataset_valid = train_test_split(dataset,test_size=config.DATA_PARAMS.TEST_SIZE, random_state=config.DATA_PARAMS.RANDOM_STATE)

    train_loader = DataLoader(dataset_train, batch_size=config.HYPER_PARAMS.BATCH_SIZE)
    val_loader = DataLoader(dataset_valid, batch_size=config.HYPER_PARAMS.BATCH_SIZE)


    model = LitClusterNet(dataset.__getitem__(0)[0].shape[0],config)

    logger = TensorBoardLogger(name="logs",save_dir=config.PATHS.SAVE_PATH)
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


    trainer.fit(model, train_loader, val_loader,)









if __name__=='__main__':

    generalised_trainer_TF_clusters()
