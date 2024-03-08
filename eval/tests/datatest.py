import sys

import torch
from torch import nn
import numpy as np

from networks.pytorch.nn_lightning import LitClusterConvolutionalNet, LitClusterNet
from tpcutils.dataset_pt import TPCTreeCluster
from tpcutils.data import SeparatedDataHandler,read_MC_tracks,select_tpc_clusters_idx
from sklearn.model_selection import train_test_split

import glob
import yaml

from matplotlib import pyplot as plt

from config.paths import dpaths as dp
from dotmap import DotMap
from scipy.stats import gaussian_kde
import ROOT

import argparse

import mplhep as hep
hep.style.use(hep.style.ALICE)

from array import array
from ROOT import addressof

from tpcio.TreeIO import create_arrays, write_ROOT_TREE

path_cosmos = '/Users/joachimcarlokristianhansen/st_O2_ML_SC_DS/TPC-analyzer/TPCTracks/models/cosmos'

pSel = 'FNet_0007_iniRef_Za0_Tgla0_dz_positiveshift'
TreeName = pSel
config = DotMap(yaml.safe_load(open('/Users/joachimcarlokristianhansen/st_O2_ML_SC_DS/TPC-analyzer/TPCTracks/py_dir/config/config_file.yml')))
configs = DotMap(yaml.safe_load(open(path_cosmos + '/' + f'{pSel}/hyperparams.yml')))

# valid
file_valid = ROOT.TFile.Open(config.PATHS.DATA_PATH_VALID)
dataset_valid = TPCTreeCluster(file_valid,transform=True,conf=configs)

y = select_tpc_clusters_idx(configs.DATA_PARAMS.TPC_SETTINGS,27-1)

