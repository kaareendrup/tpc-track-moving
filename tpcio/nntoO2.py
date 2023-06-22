import sys
from networks.pytorch.nn_lightning import LitClusterNet
import torch
import glob

import argparse


def GetOut():

    save_path = '/Users/joachimcarlokristianhansen/st_O2_ML_SC_DS/TPC-analyzer/TPCTracks/models/onnx_m'

    Net = LitClusterNet.load_from_checkpoint(glob.glob('/Users/joachimcarlokristianhansen/st_O2_ML_SC_DS/TPC-analyzer/TPCTracks/models/aurora/FNet_2/*.ckpt')[0])
    Net.eval()

    shape = 7 + 10*5

    x = torch.rand((shape))

    
    torch.onnx.export(Net,                     # model being run
                  ##since model is in the cuda mode, input also need to be
                  x.to("cpu"),              # model input (or a tuple for multiple inputs)
                  save_path + '/' + 'FNet_2_10Clusters.onnx', # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                                'output' : {0 : 'batch_size'}})




if __name__=='__main__':
    GetOut()
