import sys, os

import ROOT
from array import array
from ROOT import addressof
import numpy as np


def create_arrays(tree,var=None):

    my_var = array('f',[0])
    tar_var = array('f',[0])

    tree.Branch("NN_{}".format(var), my_var, '<float>')
    tree.Branch("MovTrackRef_{}".format(var), tar_var, '<float>')

    return my_var, tar_var

def create_iniArr(tree,var=None):

    ini_var = array('f',[0])

    tree.Branch("iniTrackRef_{}".format(var),ini_var, '<float>')
    return ini_var

def create_restArr(tree,total_var=None):

    var = array('f',[0])

    tree.Branch(f"{total_var}",var, '<float>')

    return var

def write_ROOT_TREE(tar,pred,ini,dz,imposedTB,tree_name='FNet'):
    file = ROOT.TFile.Open(f"TPC-SCD-NN-Prediction-{tree_name}.root", "RECREATE")

    tree = ROOT.TTree("tpc","tree")

    myY, tarY = create_arrays(tree,"Y")
    myZ, tarZ = create_arrays(tree,"Z")
    myPhi, tarPhi = create_arrays(tree,"phi")
    myLambda, tarLambda = create_arrays(tree,"Lambda")
    myqPt, tarqPt = create_arrays(tree,"q2pt")

    iniY = create_iniArr(tree,"Y")
    iniZ = create_iniArr(tree,"Z")
    iniPhi = create_iniArr(tree,"phi")
    iniLambda = create_iniArr(tree,"Lambda")
    iniqPt = create_iniArr(tree,"q2pt")

    var_imposedTB = create_restArr(tree,"imposedTB")
    var_dz = create_restArr(tree,"dz")


    for i in range(pred.shape[0]):

        myY[0] = float(pred[i,0])
        myZ[0] = float(pred[i,1])
        myPhi[0] = float(pred[i,2])
        myLambda[0] = float(pred[i,3])
        myqPt[0] = float(pred[i,4])


        tarY[0] = float(tar[i,0])
        tarZ[0] = float(tar[i,1])
        tarPhi[0] = float(tar[i,2])
        tarLambda[0] = float(tar[i,3])
        tarqPt[0] = float(tar[i,4])

        iniY[0] = float(ini[i,0])
        iniZ[0] = float(ini[i,1])
        iniPhi[0] = float(ini[i,2])
        iniLambda[0] = float(ini[i,3])
        iniqPt[0] = float(ini[i,4])

        var_imposedTB[0] = float(imposedTB[i])
        var_dz[0] = float(dz[i])

        tree.Fill()

    tree.Write()

    return 0
