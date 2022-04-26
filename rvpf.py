#%%
import numpy as np
import matplotlib.pyplot as plt
from cicTools import *
from scipy import spatial
import configparser
#%%
config = configparser.ConfigParser()

seed = float(config['PARAMS']['seed']) #random seed
lbox = float(config['PARAMS']['lbox']) #length of box
ngxs = float(config['PARAMS']['ngxs']) #num of galaxies
zspace = bool(config['PARAMS']['rspace']) #redshift space
zspaceAxis = config['PARAMS']['rspaceAxis'] #r-space axis
nesf = float(config['PARAMS']['nesf']) #num of test spheres

gxs = readTNG()
np.random.seed(seed)
ids = np.random.choice(len(gxs),size=ngxs)
gxs = gxs[ids]

if zspace == True:
    H0 = 67.74
    axis = zspaceAxis
    vaxis = 'v'+axis
    gxs[axis]+=gxs[vaxis]/H0
    gxs[axis][np.where(gxs[axis]<0.)[0]]+=lbox
    gxs[axis][np.where(gxs[axis]>lbox)[0]]-=lbox

pos = np.column_stack((gxs['x'],gxs['y'],gxs['z']))

tree = spatial.cKDTree(pos)
#%%

"""
Voy probando rangos de radio para calcular chi
Radio mínimo: tal que en el eje x (xi*Nmean) me de alrededor de 0.1
Radio maximo: tal que la chi no me de -inf
Estos depende del tamaño de la muestra (ngxs)

rs range for ngxs=10000: np.geomspace(1500,16000,x)
rs range for ngxs=100000: np.geomspace(500,9000,x)
rs range for ngxs=1000000: np.geomspace(200,5000,x)
"""

if ngxs==1000000: rs = np.geomspace(200,5000,20)
elif ngxs==100000: rs = np.geomspace(500,9000,20)
elif ngxs==10000: rs = np.geomspace(1500,16000,20)


P0 = np.zeros(len(rs))
N_mean = np.zeros(len(rs))
xi_mean = np.zeros(len(rs))

for i,r in enumerate(rs):
    P0[i], N_mean[i], xi_mean[i] = cic_stats(tree, nesf, r, lbox)
    

if zspace==True: 
    namefile = f'../data/ngxs{ngxs}_nesf{nesf}_redshift{axis}.npz'
else: 
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
