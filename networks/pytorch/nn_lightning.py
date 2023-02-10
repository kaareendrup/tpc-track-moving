import sys

import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau

from networks.pytorch.nn_utils import FcNet


class LitClusterNet(pl.LightningModule):
    def __init__(self,input_shape,config,**kwargs):
        super().__init__()

        self.net = FcNet(input_shape,config.MODEL.OUTPUT_SHAPE)

        self.LEARNING_RATE = config.HYPER_PARAMS.LEARNING_RATE

        self.lr_scheduler_patience = config.HYPER_PARAMS.LR_SCHEDULER_PATIENCE
        self.lr_scheduler_factor = config.HYPER_PARAMS.LR_SCHEDULER_FACTOR_

        self.save_hyperparameters()

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=self.LEARNING_RATE,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=self.lr_scheduler_patience,
            verbose=True,
            factor=self.lr_scheduler_factor
        )
        return {
           'optimizer': optimizer,
           'scheduler': scheduler,
           'monitor': 'val_loss'
       }

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)

        loss = F.mse_loss(logits, y)
        self.log('train_loss', loss,on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)

        loss = F.mse_loss(logits, y)

        self.log('val_loss', loss,on_step=False,on_epoch=True,prog_bar=True,logger=True)
