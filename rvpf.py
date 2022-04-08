#%%
import numpy as np
import matplotlib.pyplot as plt
from cicTools import *
from scipy import spatial
import random 

lbox = 205000
#V = lbox**3

N_tot = 100000

gxs = readTNG()
np.random.seed(0)
ids = np.random.choice(len(gxs),size=N_tot)
gxs = gxs[ids]
ntot = len(gxs)

#%%
H0 = 67.74
axis = 'z'
vaxis = 'vz'
gxs[axis]+=gxs[vaxis]/H0
gxs[axis][np.where(gxs[axis]<0.)[0]]+=lbox
gxs[axis][np.where(gxs[axis]>lbox)[0]]-=lbox

pos = np.column_stack((gxs['x'],gxs['y'],gxs['z']))

tree = spatial.cKDTree(pos)

rs = np.geomspace(1000,8000,5)
n = 100000

P0 = np.zeros(len(rs))
N_mean = np.zeros(len(rs))
xi_mean = np.zeros(len(rs))

for i,r in enumerate(rs):
    P0[i], N_mean[i], xi_mean[i] = cic_stats(tree, n, r, lbox)

chi = -np.log(P0)/N_mean
NE = N_mean*xi_mean

x = np.geomspace(1E-3,1E3,50)
plt.plot(x,np.log(1+x)/x)
plt.scatter(NE,chi)
plt.xscale('log')
#plt.yscale('log')
plt.show()
#%%
"""
RANDOMS
"""
lbox = 205000
ran_tree = spatial.cKDTree(lbox*np.random.rand(N_tot,3))

P0_ran = np.zeros(len(rs))
Nmean_ran = np.zeros(len(rs))
ximean_ran = np.zeros(len(rs))

for i,r in enumerate(rs):
    P0_ran[i], Nmean_ran[i], ximean_ran[i] = cic_stats(ran_tree, n, r, lbox)

chi_ran = -np.log(P0_ran)/Nmean_ran
NE_ran = Nmean_ran*ximean_ran
#%%
x = np.geomspace(1E-3,1E3,50)
plt.plot(x,np.log(1+x)/x)
plt.scatter(NE_ran,chi_ran)
plt.xscale('log')
#plt.yscale('log')
plt.show()
# %%
