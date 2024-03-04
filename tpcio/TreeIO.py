import sys, os

import ROOT
from array import array
from ROOT import addressof
import numpy as np


def create_arrays(tree,var=None):

    my_var = array('f',[0])
    tar_var = array('f',[0])
    ini_var = array('f',[0])

    tree.Branch("NN_{}".format(var), my_var, '<float>')
    tree.Branch("movTrackRef_{}".format(var), tar_var, '<float>')
    tree.Branch("iniTrackRef_{}".format(var), ini_var, '<float>')

    return my_var, tar_var, ini_var


def create_SingleArr(tree,total_var=None):

    var = array('f',[0])

    tree.Branch(f"{total_var}",var, '<float>')

    return var

def create_vector(tree,total_var=None):

    var = ROOT.std.vector('float')()  
    tree.Branch(f"{total_var}",var)

    return var

def write_ROOT_TREE(tar,pred,preds_normed,ini,mov,clusters,cluster_idx,dz,imposedTB,tree_name='FNet'):
    file = ROOT.TFile.Open(f"trees/TPC-SCD-NN-Prediction-{tree_name}.root", "RECREATE")

    tree = ROOT.TTree("tpc","tree")

    data_dic = {}
    # cleaning everything up
    name_space = ['Y','Z','Snp','Tgl','Q2Pt']
    track_types = ['NN','target','tNN','movTrackRef','iniTrackRef']
    data_grouped = [pred,tar,preds_normed,mov,ini]
    out_dic = {}
    for ii,track_key in enumerate(track_types):
        for jj,name_key in enumerate(name_space):
            out_dic[f'{track_key}_{name_key}'] = create_SingleArr(tree,f"{track_key}_{name_key}")

            data_dic[f'{track_key}_{name_key}'] = data_grouped[ii][:,jj]


    var_imposedTB = create_SingleArr(tree,"imposedTB")
    var_dz = create_SingleArr(tree,"dz")

    # ini_clX = create_vector(tree,"ini_clX")
    cluster_names = ['ini_clX','ini_clY','ini_clZ','mov_clX','mov_clY','mov_clZ','ini_clSector','ini_clRow']
    vector_var = []
    for name in cluster_names:
        vector_var.append(create_vector(tree,name))
    

    print("pred shape",pred.shape[0])
    print("#"*25)
    
    for i in range(pred.shape[0]):
        sys.stdout.write("\rprocessing %i/%i : %.2f%%" % (i+1,pred.shape[0],(i+1)/pred.shape[0]*100))
        sys.stdout.flush()

        for track_key in track_types:
            for name_key in name_space:
                out_dic[f"{track_key}_{name_key}"][0] = float(data_dic[f"{track_key}_{name_key}"][i])

        
        for typeCluster in range(len(clusters[i])):
            for nClusters in cluster_idx[i]:
                vector_var[typeCluster].push_back(clusters[i][typeCluster][int(nClusters)])
        

        var_imposedTB[0] = float(imposedTB[i])
        var_dz[0] = float(dz[i])

        tree.Fill()

        for typeCluster in range(len(clusters[i])):
            vector_var[typeCluster].clear()


    tree.Write()

