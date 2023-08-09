import sys

import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau

from networks.pytorch.nn_networks import FcNet, DeepConvSimpleNet, PseudoGraph, PseudoGraphSinglePhi, PseudoGraphSingleLambda, PseudoGraphSingleLinear ,mRNN
from tpcutils.training_pt import PiecewiseLinearLR, LogCoshLoss, VonMisesFisher2DLoss, eps_like
from torch.optim import Adam


class LitClusterNet(pl.LightningModule):
    def __init__(self,input_shape,config,**kwargs):
        super().__init__()

        self.net = FcNet(input_shape,config.MODEL.OUTPUT_SHAPE)

        self.LEARNING_RATE = config.HYPER_PARAMS.LEARNING_RATE

        self.lr_scheduler_patience = config.HYPER_PARAMS.LR_SCHEDULER_PATIENCE
        self.lr_scheduler_factor = config.HYPER_PARAMS.LR_SCHEDULER_FACTOR_

        self.save_hyperparameters()

        #self._loss1 = LogCoshLoss()
        # self._loss1 = nn.MSELoss()
        # self._loss2 = VonMisesFisher2DLoss()
        self._lossT = nn.MSELoss()

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
        #linloss = self._loss1( torch.cat((logits[...,:2], logits[...,4].unsqueeze(1)),dim=1) , torch.cat((y[...,:2],y[...,4].unsqueeze(1)),dim=1) )
        #angloss = self._loss2( torch.cat((logits[:,2].unsqueeze(1), logits[:,3].unsqueeze(1)), dim=1).float(), torch.cat((y[...,2].unsqueeze(1),y[...,3].unsqueeze(1)),dim=1).float())
        # loss = torch.cat((linloss, angloss.unsqueeze(1)), dim=1).mean()
        # logits[:,2] = torch.sin(logits[:,2])
        # logits[:,3] = torch.tan(logits[:,3])
        
        # nLogits = torch.cat((logits[:,0].unsqueeze(1),logits[:,1].unsqueeze(1),snp.unsqueeze(1),tgl.unsqueeze(1),logits[:,4].unsqueeze(1)),dim=1)


        loss = self._lossT(logits,y)

        self.log('train_loss', loss,on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x.to(self.device)
        y.to(self.device)
        logits = self.forward(x)

        # logits: Y:0 Z:1 Phi:2 Tgl:3 q2:4
        # y: Y:0 Z:1 Phi:2 Tgl:3 q2:4
        # linloss = self._loss1( torch.cat((logits[...,:2], logits[...,4].unsqueeze(1)),dim=1) , torch.cat((y[...,:2],y[...,4].unsqueeze(1)),dim=1) )
        # angloss = self._loss2( torch.cat((logits[:,2].unsqueeze(1), logits[:,3].unsqueeze(1)), dim=1).float(), torch.cat((y[...,2].unsqueeze(1),y[...,3].unsqueeze(1)),dim=1).float())
        # loss = torch.cat((linloss, angloss.unsqueeze(1)), dim=1).mean()
        #Y,Z,Snp,Tgl,q2pt = logits # and batch
        # logits[:,2] = torch.sin(logits[:,2])
        # logits[:,3] = torch.tan(logits[:,3])


        loss = self._lossT(logits,y)
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

        # Add 2 to output shape to predict kappa for double vMF loss
        self.net = PseudoGraph(input_shape,config.MODEL.OUTPUT_SHAPE+2)

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
        self._loss3 = VonMisesFisher2DLoss()

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

        linloss = self._loss1( torch.cat((logits[:,:2],logits[:,4:5]), dim=1) , torch.cat((y[:,:2],y[:,4:]), dim=1) )
        # Scale Z loss
        linloss[:,1] /= 10
        
        angloss_phi = torch.abs(
            self._loss2(
                torch.cat((
                    torch.asin(logits[:,2].unsqueeze(1)), 
                    logits[:,5].unsqueeze(1) + eps_like(logits[:,5].unsqueeze(1)),
                ), dim=1), 
                torch.asin(y[:,2]).unsqueeze(1) 
            )
        ) * 8

        angloss_lambda = torch.log(torch.cosh(
            self._loss3(
                torch.cat((
                    torch.atan(logits[:,3].unsqueeze(1)), 
                    logits[:,6].unsqueeze(1) + eps_like(logits[:,6].unsqueeze(1)),
                ), dim=1), 
                torch.atan(y[:,3]).unsqueeze(1)
            )
        )) * 8
        
        print(torch.cat((linloss, angloss_phi.unsqueeze(1), angloss_lambda.unsqueeze(1)), dim=1).mean(dim=0))
        loss = torch.cat((linloss, angloss_phi.unsqueeze(1), angloss_lambda.unsqueeze(1)), dim=1).mean()
        
        if loss > 10:
            print(torch.cat((linloss, angloss_phi.unsqueeze(1), angloss_lambda.unsqueeze(1)), dim=1))

        if torch.any(torch.isnan(loss)):
            print('Found nan loss!')
            print(torch.cat((linloss, angloss_phi.unsqueeze(1), angloss_lambda.unsqueeze(1)), dim=1))

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)

        # New double angloss
        linloss = self._loss1( torch.cat((logits[:,:2],logits[:,4:5]), dim=1) , torch.cat((y[:,:2],y[:,4:]), dim=1) )
        # Scale Z loss
        linloss[:,1] /= 10

        angloss_phi = torch.abs(
            self._loss2(
                torch.cat((
                    torch.asin(logits[:,2].unsqueeze(1)), 
                    logits[:,5].unsqueeze(1) + eps_like(logits[:,5].unsqueeze(1)),
                ), dim=1), 
                torch.asin(y[:,2]).unsqueeze(1) 
            )
        ) * 8

        angloss_lambda = torch.log(torch.cosh(
            self._loss3(
                torch.cat((
                    torch.atan(logits[:,3].unsqueeze(1)), 
                    logits[:,6].unsqueeze(1) + eps_like(logits[:,6].unsqueeze(1)),
                ), dim=1), 
                torch.atan(y[:,3]).unsqueeze(1)
            )
        )) * 8
        
        loss = torch.cat((linloss, angloss_phi.unsqueeze(1), angloss_lambda.unsqueeze(1)), dim=1).mean()

        if loss > 10:
            print(torch.cat((linloss, angloss_phi.unsqueeze(1), angloss_lambda.unsqueeze(1)), dim=1))

        if torch.any(torch.isnan(loss)):
            torch.set_printoptions(threshold=10_000)
            print(' ')
            print('Found nan values!')
            print('logits')
            print(logits.size())
            print(logits)
            print('phi')
            print(torch.asin(logits[:,2].unsqueeze(1)))
            print('lambda')
            print(torch.atan(logits[:,3].unsqueeze(1)))
            print('kappas')
            print(logits[:,5].unsqueeze(1) + eps_like(x))
            print(logits[:,6].unsqueeze(1) + eps_like(x))
            print('loss')
            print(torch.cat((linloss, angloss_phi.unsqueeze(1), angloss_lambda.unsqueeze(1)), dim=1))

        self.log('val_loss', loss,on_step=False,on_epoch=True,prog_bar=True,logger=True)


class PseudoGraphNetSingle(pl.LightningModule):

    def __init__(self,input_shape,config,training_dataloader,which,**kwargs):
        super().__init__()

        self._which = which
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

        # Add 1 to output shape to predict kappa for double vMF loss
        if self._which == 2:
            print('Using PGSPhi')
            self.net = PseudoGraphSinglePhi(input_shape,config.MODEL.OUTPUT_SHAPE+1)
            self._loss = VonMisesFisher2DLoss()

        elif self._which == 3:
            print('Using PGSLambda')
            self.net = PseudoGraphSingleLambda(input_shape,config.MODEL.OUTPUT_SHAPE+1)
            self._loss = VonMisesFisher2DLoss()

        else:
            self.net = PseudoGraphSingleLinear(input_shape,config.MODEL.OUTPUT_SHAPE)
            self._loss = LogCoshLoss()

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
        logits[:,-1] += eps_like(logits[:,-1].unsqueeze(1))

        loss = torch.mean(
            torch.abs(
                self._loss(logits, y[:,self._which].unsqueeze(1))
            )
        )

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        logits[:,-1] += eps_like(logits[:,-1].unsqueeze(1))

        loss = torch.mean(
            torch.abs(
                self._loss(logits, y[:,self._which].unsqueeze(1))
            )
        )

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