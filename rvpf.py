#%%
import numpy as np
import matplotlib.pyplot as plt
from cicTools import *
from scipy import spatial
import random 

lbox = 205000

ngxs = 1000000

gxs = readTNG()
np.random.seed(0)
ids = np.random.choice(len(gxs),size=ngxs)
gxs = gxs[ids]
#ntot = len(gxs)

# H0 = 67.74
# axis = 'y'
# vaxis = 'v'+axis
# gxs[axis]+=gxs[vaxis]/H0
# gxs[axis][np.where(gxs[axis]<0.)[0]]+=lbox
# gxs[axis][np.where(gxs[axis]>lbox)[0]]-=lbox

pos = np.column_stack((gxs['x'],gxs['y'],gxs['z']))

tree = spatial.cKDTree(pos)
#%%

rs = np.geomspace(100,6000,20)
nesf = 100000

P0 = np.zeros(len(rs))
N_mean = np.zeros(len(rs))
xi_mean = np.zeros(len(rs))

for i,r in enumerate(rs):
    P0[i], N_mean[i], xi_mean[i] = cic_stats(tree, nesf, r, lbox)
    

namefile = f'../data/ngxs{ngxs}_nesf{nesf}.npz'
np.savez(namefile,P0,N_mean,xi_mean,rs)

x = np.geomspace(1E-2,1E2,50)
c='k'
chi = -np.log(P0)/N_mean
NE = N_mean*xi_mean

plt.plot(x,np.log(1+x)/x,label='Negative Binomial',c=c)
a=.3
plt.plot(x,(1/((1-a)*(x/a)))*((1+x/a)**(1-a)-1),label='Generalized Hierarhichal',c=c,ls='--')
#plt.plot(x,(1-np.e**(-x))/x,label='Minimal')
plt.plot(x,(np.sqrt(1+2*x)-1)/x,label='Thermodynamical',c=c,ls='-.')
#plt.plot(x[:-15],1-x[:-15]/2,label='Gauss',c=c)
# Q=1
# plt.plot(x,1-(np.euler_gamma+np.log(4*Q*x))/(8*Q),label='BBGKY',c=c,ls=':')

plt.scatter(NE,chi,lw=2)
plt.xscale('log')
plt.legend(loc=3)
plt.show()
# %%
