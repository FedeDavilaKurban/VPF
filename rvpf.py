#%%
import numpy as np
import matplotlib.pyplot as plt
from cicTools import *
from scipy import spatial
import random 

lbox = 205000
V = lbox**3

gxs = readTNG()
ids = np.random.choice(len(gxs),size=10000)
gxs = gxs[ids]
ntot = len(gxs)

pos = np.column_stack((gxs['x'],gxs['y'],gxs['z']))

tree = spatial.cKDTree(pos)

#%%

rs = np.linspace(10000,68000,15)
n = 100000 

P0 = np.zeros(len(rs))
N_mean = np.zeros(len(rs))
xi_mean = np.zeros(len(rs))

for i,r in enumerate(rs):
    P0[i], N_mean[i], xi_mean[i] = cic_stats(tree, n, r, seed=101)

chi = -np.log(P0)/N_mean
NE = N_mean*xi_mean
#%%
x = np.geomspace(1E-3,1E3,50)
plt.plot(x,np.log(1+x)/x)
plt.scatter(NE,chi)
plt.xscale('log')
#plt.yscale('log')
plt.show()
# %%
"""
RANDOMS
"""
nran = 10000
ran_tree = spatial.cKDTree(np.random.rand(nran,3))
#rs = np.linspace(10000,68000,15)
#n = 100000 

P0 = np.zeros(len(rs))
N_mean = np.zeros(len(rs))
xi_mean = np.zeros(len(rs))

for i,r in enumerate(rs):
    P0[i], N_mean[i], xi_mean[i] = cic_stats(ran_tree, n, r, seed=101)
#%%
chi = -np.log(P0)/N_mean
NE = N_mean*xi_mean

x = np.geomspace(1E-3,1E3,50)
plt.plot(x,np.log(1+x)/x)
plt.scatter(NE,chi)
plt.xscale('log')
#plt.yscale('log')
plt.show()
# %%
