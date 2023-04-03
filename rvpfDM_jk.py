#%%
import numpy as np
import matplotlib.pyplot as plt
from cicTools import *
from scipy import spatial
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

seed = int(config['PARAMS']['seed']) #random seed
lbox = float(config['PARAMS']['lbox']) #length of box
ngxs = float(config['PARAMS']['ngxs']) #dilution
zspace = config['PARAMS'].getboolean('zspace') #redshift space
zspaceAxis = config['PARAMS']['zspaceAxis'] #r-space axis
nesf = int(config['PARAMS']['nesf']) #num of test spheres
rsbin = int(config['PARAMS']['rsbin']) #num of bins of r
jk = int(config['PARAMS']['jk']) #num of bins of r
invoid = config['PARAMS'].getboolean('invoid') #redshift space
#ompleteRrange = config['PARAMS'].getboolean('completeRrange')

print(f"""
      ngxs = {ngxs}
      nesf = {nesf}
      zspace = {zspace}
      zspaceAxis = {zspaceAxis}
      Num of JK resamplings = {jk}^3
      invoid = {invoid}
      """)
      #completeRrange = {completeRrange}

#%%
rs = np.geomspace(40,4000,rsbin) 

DM = np.load('../data/dmpos_diluted.npz')
pos = DM['arr_0']
vel = DM['arr_1']
del DM

if zspace == True:
    H0 = .06774
    if zspaceAxis == 'x':
        pos[:,0]+=vel[:,0]/H0
        pos[:,0][np.where(pos[:,0]<0.)[0]]+=lbox
        pos[:,0][np.where(pos[:,0]>lbox)[0]]-=lbox
    elif zspaceAxis == 'y':
        pos[:,1]+=vel[:,1]/H0
        pos[:,1][np.where(pos[:,1]<0.)[0]]+=lbox
        pos[:,1][np.where(pos[:,1]>lbox)[0]]-=lbox
    elif zspaceAxis == 'z':
        pos[:,2]+=vel[:,2]/H0
        pos[:,2][np.where(pos[:,2]<0.)[0]]+=lbox
        pos[:,2][np.where(pos[:,2]>lbox)[0]]-=lbox


tree = spatial.cKDTree(pos)

chi = np.zeros(len(rs))
NXi = np.zeros(len(rs))
P0 = np.zeros(len(rs))
N_mean = np.zeros(len(rs))
xi_mean = np.zeros(len(rs))

if invoid == False:
    if jk!= 0:
        chi_std = np.zeros(len(rs))
        NXi_std = np.zeros(len(rs))
        P0_std = np.zeros(len(rs))
        N_mean_std = np.zeros(len(rs))
        xi_mean_std = np.zeros(len(rs))

        print('Calculating JK cic statistics...')
        
        for i,r in enumerate(rs):
            chi[i], NXi[i], P0[i], N_mean[i], xi_mean[i],\
                    chi_std[i], NXi_std[i], P0_std[i], N_mean_std[i], xi_mean_std[i]\
                        = cic_stats_jk(tree, nesf, r, lbox, jk)
    else:
        print('Calculating cic statistics...')
        for i,r in enumerate(rs):
            chi[i], NXi[i], P0[i], N_mean[i], xi_mean[i],\
                        = cic_stats(tree, nesf, r, lbox)

if invoid == True:
    if jk==3:
        chi_std = np.zeros(len(rs))
        NXi_std = np.zeros(len(rs))
        P0_std = np.zeros(len(rs))
        N_mean_std = np.zeros(len(rs))
        xi_mean_std = np.zeros(len(rs))

        print('Calculating JK invoid cic statistics...')
        
        for i,r in enumerate(rs):
            chi[i], NXi[i], P0[i], N_mean[i], xi_mean[i],\
                    chi_std[i], NXi_std[i], P0_std[i], N_mean_std[i], xi_mean_std[i]\
                        = cic_stats_invoid_jk(tree, nesf, r)

    else:
        print('Calculating invoid cic statistics...')
        for i,r in enumerate(rs):
            chi[i], NXi[i], P0[i], N_mean[i], xi_mean[i],\
                        = cic_stats_invoid(tree, nesf, r)

##########
# Writing
##########
namefile = '../data/DMdata'
if zspace==True: 
    namefile += f'_redshift{zspaceAxis}'
if jk!=0:
    namefile += '_jk'

namefile += '.npz'
print(f'Creating {namefile}')

if jk!=0:
    np.savez(namefile,chi,chi_std,NXi,NXi_std,P0,P0_std,N_mean,N_mean_std,xi_mean,xi_mean_std,rs)
else:
    np.savez(namefile,chi,NXi,P0,N_mean,xi_mean,rs)


# %%
