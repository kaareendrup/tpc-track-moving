import sys, os

from array import array
from ROOT import addresso
import numpy as np


def create_arrays(tree,var=None):

    my_var = array('f',[0])
    tar_var = array('f',[0])

    tree.Branch("NN_{}".format(var), my_var, '<float>')
    tree.Branch("MovTrackRef_{}".format(var), tar_var, '<float>')

    return my_var, tar_var

def write_ROOT_TREE(tar,pred,tree_name='FNet'):
    file = ROOT.TFile.Open(f"TPC-SCD-NN-Prediction-{tree_name}.root", "RECREATE")

    tree = ROOT.TTree("tpc","tree")

    myY, tarY = create_arrays(tree,"Y")
    myZ, tarZ = create_arrays(tree,"Z")
    myPhi, tarPhi = create_arrays(tree,"phi")
    myLambda, tarLambda = create_arrays(tree,"Lambda")
    myqPt, tarqPt = create_arrays(tree,"q2pt")



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

        tree.Fill()

    tree.Write()

    return 0
