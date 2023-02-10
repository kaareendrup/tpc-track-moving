import sys,os
import numpy as np
import pandas as pd
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import axes3d
import matplotlib.animation as animation

from matplotlib import pyplot as plt

import glob

import argparse

def main(args):

    data_path = '/Users/joachimcarlokristianhansen/st_O2_ML_SC_DS/TPC-analyzer/TPCTracks/data_files'

    # iEntry << " " << mX << " " << mAlpha << " " << mY << " " << mZ << " " << Snp << " " << Tgl << " " << Q2Pt << " " << bcTB << " " << dz << " ";
    data_names = ["X","alpha","Y","Z","sin_phi","tgLambda","q2pt","bcTB","dz","cov1","cov2","cov3","cov4","cov5","cov6","cov7","cov8",
                     "cov9","cov10","cov11","cov12","cov13","cov14","cov15"]


    files = glob.glob(data_path + '/*.txt')
    for fi in files:
        temp = fi.split('/')[-1]
        # print(temp)


    iniTrack = pd.read_csv(files[1],header=None,sep=' ',index_col=0)#names=data_names)
    iniTrackRef = pd.read_csv(files[0],header=None,sep=' ',index_col=0)#names=data_names)
    movTrackRef = pd.read_csv(files[2],header=None,sep=' ',index_col=0)#,names=data_names)
    mcTrack = pd.read_csv(files[3],header=None,sep=' ',index_col=0)#names=data_names)

    nClusters = 159

    ini = iniTrack.values[:,:-1]
    iniRef = iniTrackRef.values[:,:-1]
    mov = movTrackRef.values[:,:-1]
    mc = mcTrack.values[:,:-1]

    def get_XY(array,i):
        #array = ini[0,:][len(data_names)+nClusters*2:].reshape(159,3)
        X0 = array[i,:][len(data_names)+(nClusters*2):][0::3]
        Y0 = array[i,:][len(data_names)+(nClusters*2):][1::3]
        Z0 = array[i,:][len(data_names)+(nClusters*2):][2::3]

        # remove padding
        mask = np.where(X0==0)[0][0]
        X0 = X0[:mask]
        Y0 = Y0[:mask]
        Z0 = Z0[:mask]


        # xyz = xyz_data.iloc[[track]].to_numpy()
        # xyz = np.reshape(xyz, (3,-1), order='F')
        #
        # cut = np.where(xyz==0)[1][0]
        # xyz = xyz[:,:cut]

        sector_data = array[i,:][len(data_names):len(data_names)+nClusters]
        # Correct for sector
        #sector = sector_data.iloc[[track]].to_numpy()
        sector = sector_data[:mask]
        print(sector)
        sector_corr = - sector * 20/360*2*np.pi


        x_new = X0[0] * np.cos(sector_corr) + Y0[1] * np.sin(sector_corr)
        y_new = - X0[0] * np.sin(sector_corr) + Y0[1] * np.cos(sector_corr)

        return x_new,y_new,Z0


    xyz_data=iniTrack.iloc[:,len(data_names)+nClusters*2:-1]
    sector_data = iniTrack.iloc[:,len(data_names):len(data_names)+nClusters]

    xyz_data_mov=movTrackRef.iloc[:,len(data_names)+nClusters*2:-1]
    sector_data_mov = movTrackRef.iloc[:,len(data_names):len(data_names)+nClusters]



    def get_xyz_new(xyz_data,sector_data,track):

        xyz = xyz_data.iloc[[track]].to_numpy()
        xyz = np.reshape(xyz, (3,-1), order='F')

        cut = np.where(xyz==0)[1][0]
        xyz = xyz[:,:cut]

        # Correct for sector
        sector = sector_data.iloc[[track]].to_numpy()
        sector = sector[0,:cut]
        sector_corr = - sector * 20/360*2*np.pi

        x_new = xyz[0] * np.cos(sector_corr) + xyz[1] * np.sin(sector_corr)
        y_new = - xyz[0] * np.sin(sector_corr) + xyz[1] * np.cos(sector_corr)

        return x_new,y_new, xyz[2]

    # fig = plt.figure(figsize=(8,8))
    # ax = plt.axes(projection='3d')

    #xy
    # plt.rcParams["image.cmap"] = "Set1"
    # # to change default color cycle
    # plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.tab20.colors) # discrete


    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.Reds(np.linspace(0,1,args.nTracks))) # continous

    # f = plt.figure(figsize=(8,6))

    if args.fig_type=='2D':
        f,axs = plt.subplots(1,2,figsize=(12,6))

        f.suptitle("TPC-clusters to lab coordinate view")

        ax=axs[0]

        for i in range(0,args.nTracks,1):
            # x_new,y_new = get_xyz_new(ini,i)
            # plt.plot(x_new,y_new,'-',color='purple',alpha=0.5)
            X0,Y0,_ = get_xyz_new(xyz_data,sector_data,i)
            ax.scatter(X0,Y0,s=1)

            #X_re,Y_re,_ = get_XY(iniRef,i)

            #plt.plot(X_re,Y_re,'-',color='black',alpha=0.5)

            # mX,mY,_ = get_XY(mov,i)
            # plt.plot(mX,mY,'-',color='red',alpha=0.5)

        ax.set_xlim(-250,250)
        ax.set_ylim(-250,250)
        ax.set(xlabel='x', ylabel='y')
        ax.set_title("xy-projection")
        #plt.xlim(-60,60)
        ax.set_aspect(1)
        ax=axs[1]
        #yz
        for i in range(0,args.nTracks,1):
            # x_new,y_new = get_xyz_new(ini,i)
            # plt.plot(x_new,y_new,'-',color='purple',alpha=0.5)
            _,Y0,Z0 = get_xyz_new(xyz_data,sector_data,i)
            ax.scatter(Z0,Y0,s=1,)

            #X_re,Y_re,_ = get_XY(iniRef,i)

            #plt.plot(X_re,Y_re,'-',color='black',alpha=0.5)

            # mX,mY,_ = get_XY(mov,i)
            # plt.plot(mX,mY,'-',color='red',alpha=0.5)

        ax.set_xlim(-250,250)
        ax.set_ylim(-250,250)
        ax.set(xlabel='z', ylabel='y')
        ax.set_title("zy-projection")
        ax.set_aspect(1)
        plt.show()

    elif args.fig_type=='3D':


        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # ax.plot3D(X0, Y0, Z0, 'gray')


        for i in range(0,args.nTracks,1):
            X0,Y0,Z0 = get_xyz_new(xyz_data,sector_data,i)
            # X_re,Y_re,Z_re = get_XY(iniRef,i)
            mX,mY,mZ = get_xyz_new(xyz_data_mov,sector_data_mov,i)

            #ax.plot3D(X0, Y0, Z0, 'gray')
            ax.scatter3D(X0, Y0, Z0, c=Z0, cmap='Reds',alpha=1,s=8);


            #ax.plot3D(mX, mY, mZ, 'gray')
            ax.scatter3D(mX, mY, mZ, c=mZ, cmap='Greens',alpha=0.5,s=8);


        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        ax.set_xlim(-250,250)
        ax.set_ylim(-250,250)
        ax.set_zlim(-250,250)


        # plt.show()
        if args.animate:
            # Rotate the axes and update
            for angle in range(1, 360*3 + 1):
                # Normalize the angle to the range [-180, 180] for display
                angle_norm = (angle + 180) % 360 - 180

                # Cycle through a full rotation of elevation, then azimuth, roll, and all
                elev = azim = 1
                if angle <= 360:
                    elev = angle_norm
                elif angle <= 360*2:
                    azim = angle_norm
                else:
                    elev = azim = angle_norm



                # Update the axis view and title
                ax.view_init(elev=elev, azim=azim, vertical_axis='z')
                #ax.view_init(elev, azim)
                plt.title('a: %d°, b: %d°' % (elev, azim))

                plt.draw()
                plt.pause(.001)
        else:
            plt.show()

if __name__=='__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--fig_type",
                        default="2D",
                        required=False,
                        help="2D or 3D figures"
                        )

    parser.add_argument("--nTracks",
                        default=100,
                        type = int,
                        required=False,
                        help="number of tracks plotted"
                        )

    parser.add_argument("--animate",
                        default=False,
                        type = int,
                        required=False,
                        help="show 3d as animation"
                        )


    args = parser.parse_args()

    main(args)
