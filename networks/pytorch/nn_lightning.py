import sys

import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau

from networks.pytorch.nn_networks import FcNet,DeepConvSimpleNet, PseudoGraph, mRNN
from tpcutils.training_pt import PiecewiseLinearLR, LogCoshLoss, VonMisesFisher2DLoss
from torch.optim import Adam


class LitClusterNet(pl.LightningModule):
    def __init__(self,input_shape,config,**kwargs):
        super().__init__()

        self.net = FcNet(input_shape,config.MODEL.OUTPUT_SHAPE)

        self.LEARNING_RATE = config.HYPER_PARAMS.LEARNING_RATE

        self.lr_scheduler_patience = config.HYPER_PARAMS.LR_SCHEDULER_PATIENCE
        self.lr_scheduler_factor = config.HYPER_PARAMS.LR_SCHEDULER_FACTOR_

        self.save_hyperparameters()

        self._loss1 = LogCoshLoss()
        self._loss2 = VonMisesFisher2DLoss()

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
        x.to(self.device)
        y.to(self.device)
        logits = self.forward(x)


        #loss = F.mse_loss(logits, y)
        linloss = self._loss1( torch.cat((logits[...,:2], logits[...,4].unsqueeze(1)),dim=1) , torch.cat((y[...,:2],y[...,4].unsqueeze(1)),dim=1) )
        angloss = self._loss2( torch.cat((logits[:,2].unsqueeze(1), logits[:,3].unsqueeze(1)), dim=1).float(), torch.cat((y[...,2].unsqueeze(1),y[...,3].unsqueeze(1)),dim=1).float())

        loss = torch.cat((linloss, angloss.unsqueeze(1)), dim=1).mean()

        self.log('train_loss', loss,on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x.to(self.device)
        y.to(self.device)
        logits = self.forward(x)


        linloss = self._loss1( torch.cat((logits[...,:2], logits[...,4].unsqueeze(1)),dim=1) , torch.cat((y[...,:2],y[...,4].unsqueeze(1)),dim=1) )
        angloss = self._loss2( torch.cat((logits[:,2].unsqueeze(1), logits[:,3].unsqueeze(1)), dim=1).float(), torch.cat((y[...,2].unsqueeze(1),y[...,3].unsqueeze(1)),dim=1).float())

        loss = torch.cat((linloss, angloss.unsqueeze(1)), dim=1).mean()
        #loss = F.mse_loss(logits, y)

        self.log('val_loss', loss,on_step=False,on_epoch=True,prog_bar=True,logger=True)


class LitClusterConvolutionalNet(pl.LightningModule):
    def __init__(self,config,**kwargs):
        super().__init__()

        filter = config.MODEL.FILTER

        self.net = DeepConvSimpleNet(1,config.MODEL.OUTPUT_SHAPE,filter)

        self.LEARNING_RATE = config.HYPER_PARAMS.LEARNING_RATE

        self.lr_scheduler_patience = config.HYPER_PARAMS.LR_SCHEDULER_PATIENCE
        self.lr_scheduler_factor = config.HYPER_PARAMS.LR_SCHEDULER_FACTOR_

        self.save_hyperparameters()

    def forward(self, x3vec,xmP):
        return self.net(x3vec,xmP)

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
        x3vec = train_batch['input_xyz_row']
        xmp = train_batch['mP']
        y = train_batch['target']

        logits = self.forward(x3vec,xmp)

        loss = F.mse_loss(logits, y)
        self.log('train_loss', loss,on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x3vec = val_batch['input_xyz_row']
        xmp = val_batch['mP']
        y = val_batch['target']

        logits = self.forward(x3vec,xmp)

        loss = F.mse_loss(logits, y)

        self.log('val_loss', loss,on_step=False,on_epoch=True,prog_bar=True,logger=True)


class PseudoGraphNet(pl.LightningModule):
    def __init__(self,input_shape,config,training_dataloader,**kwargs):
        super().__init__()

        # Add 1 to output shape to predict kappa for vMF loss
        self.net = PseudoGraph(input_shape,config.MODEL.OUTPUT_SHAPE+1)

        self.save_hyperparameters()

        self._optimizer_class=Adam
        self._optimizer_kwargs={'lr': config.HYPER_PARAMS.LEARNING_RATE, 'eps': config.HYPER_PARAMS.LEARNING_RATE_EPS}

        self._scheduler_class=PiecewiseLinearLR
        self._scheduler_kwargs={
            'milestones': [0, len(training_dataloader) / 2, len(training_dataloader) * config.HYPER_PARAMS.MAX_EPOCHS],
            'factors': [1e-2, 1, 1e-02]
        }
        self._scheduler_config={
            'interval': 'step',
        }

        self._loss1 = LogCoshLoss()
        self._loss2 = VonMisesFisher2DLoss()

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        optimizer = self._optimizer_class(
            self.parameters(), **self._optimizer_kwargs
        )
        config = {
            "optimizer": optimizer,
        }
        if self._scheduler_class is not None:
            scheduler = self._scheduler_class(
                optimizer, **self._scheduler_kwargs
            )
            config.update(
                {
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        **self._scheduler_config,
                    },
                }
            )
        return config

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)

        # Custom loss step
        linloss = self._loss1( torch.cat((logits[:,:2],logits[:,3:5]), dim=1) , torch.cat((y[:,:2],y[:,3:]), dim=1) )
        angloss = self._loss2( torch.cat((logits[:,2].unsqueeze(1), logits[:,5].unsqueeze(1)), dim=1), y[:,2].unsqueeze(1) )

        loss = torch.cat((linloss, angloss.unsqueeze(1)), dim=1).mean()
        self.log('train_loss', loss,on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)

        # Custom loss step
        linloss = self._loss1( torch.cat((logits[:,:2],logits[:,3:5]), dim=1) , torch.cat((y[:,:2],y[:,3:]), dim=1) )
        # angloss = self._loss2( torch.cat((logits[:,2].unsqueeze(1), logits[:,5].unsqueeze(1)), dim=1), y[:,2].unsqueeze(1) )
        angloss = self._loss2( torch.cat((logits[:,2], logits[:,5]), dim=0), torch.cat((y[:,2],y[:,3]),dim=0) )
        print("angloss",angloss.shape)
        loss = torch.cat((linloss, angloss.unsqueeze(1)), dim=1).mean()
        self.log('val_loss', loss,on_step=False,on_epoch=True,prog_bar=True,logger=True)


class LitRNN(pl.LightningModule):
    def __init__(self,input_shape,config,**kwargs):
        super().__init__()

        self.net = mRNN(input_size=input_shape, output_size=config.MODEL.OUTPUT_SHAPE, hidden_dim=50, n_layers=4)

        self.LEARNING_RATE = config.HYPER_PARAMS.LEARNING_RATE

        self.lr_scheduler_patience = config.HYPER_PARAMS.LR_SCHEDULER_PATIENCE
        self.lr_scheduler_factor = config.HYPER_PARAMS.LR_SCHEDULER_FACTOR_

        self.save_hyperparameters()

    def forward(self, x):
        return self.net(x,self.device)

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
        x = x.unsqueeze(0)
        logits,hn = self.forward(x)
        
        loss = F.mse_loss(logits, y)
        self.log('train_loss', loss,on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.unsqueeze(0)
        logits,hn = self.forward(x)

        loss = F.mse_loss(logits, y)

        self.log('val_loss', loss,on_step=False,on_epoch=True,prog_bar=True,logger=True)