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
ngxs = int(config['PARAMS']['ngxs']) #num of galaxies
zspace = config['PARAMS'].getboolean('zspace') #redshift space
zspaceAxis = config['PARAMS']['zspaceAxis'] #r-space axis
nesf = int(config['PARAMS']['nesf']) #num of test spheres
rsbin = int(config['PARAMS']['rsbin']) #num of bins of r of the spheres

print(f"""
      ngxs = {ngxs}
      nesf = {nesf}
      zspace = {zspace}
      zspaceAxis = {zspaceAxis}
      """)


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
    if zspaceAxis == 'y':
        pos[:,1]+=vel[:,1]/H0
        pos[:,1][np.where(pos[:,1]<0.)[0]]+=lbox
        pos[:,1][np.where(pos[:,1]>lbox)[0]]-=lbox
    if zspaceAxis == 'z':
        pos[:,2]+=vel[:,2]/H0
        pos[:,2][np.where(pos[:,2]<0.)[0]]+=lbox
        pos[:,2][np.where(pos[:,2]>lbox)[0]]-=lbox


tree = spatial.cKDTree(pos)

chi = np.zeros(len(rs))
NXi = np.zeros(len(rs))
P0 = np.zeros(len(rs))
N_mean = np.zeros(len(rs))
xi_mean = np.zeros(len(rs))

for i,r in enumerate(rs):
    chi[i], NXi[i], P0[i], N_mean[i], xi_mean[i] = cic_stats(tree, nesf, r, lbox)

##########
# Writing
##########
namefile = 'DMdata'
if zspace==True: 
    namefile += f'_redshift{zspaceAxis}'
namefile += '.npz'
print(f'Creating {namefile}')
np.savez(namefile,P0,N_mean,xi_mean,rs)

#%%
##########
# Reading
##########
# import numpy as np

# data = np.load('DMdata.npz')

# P0=data['arr_0']
# N_mean = data['arr_1']
# xi_mean = data['arr_2']
# rs = data['arr_3']

#%%
##########
# Plotting
##########
import matplotlib.pyplot as plt
x = np.geomspace(1E-2,1E3,50)
c='k'
chi = -np.log(P0)/N_mean
NE = N_mean*xi_mean

plt.plot(x,np.log(1+x)/x,label='Negative Binomial',c=c)
a=.2
plt.plot(x,(1/((1-a)*(x/a)))*((1+x/a)**(1-a)-1),label='Generalized Hierarhichal',c=c,ls='--')
#plt.plot(x,(1-np.e**(-x))/x,label='Minimal')
plt.plot(x,(np.sqrt(1+2*x)-1)/x,label='Thermodynamical',c=c,ls='-.')
#plt.plot(x[:-15],1-x[:-15]/2,label='Gauss',c=c)
# Q=1
# plt.plot(x,1-(np.euler_gamma+np.log(4*Q*x))/(8*Q),label='BBGKY',c=c,ls=':')

plt.scatter(NE,chi,lw=2)
plt.xscale('log')
plt.legend(loc=3)
plt.savefig(f'../plots/test_rvpfDM_{namefile}.png')
# %%
