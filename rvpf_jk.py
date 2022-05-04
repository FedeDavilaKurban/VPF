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
rsbin = int(config['PARAMS']['rsbin']) #num of bins of r
jk = int(config['PARAMS']['jk']) #num of bins of r
invoid = config['PARAMS'].getboolean('invoid') #redshift space

print(f"""
      ngxs = {ngxs}
      nesf = {nesf}
      zspace = {zspace}
      zspaceAxis = {zspaceAxis}
      Num of JK resamplings = {jk}^3
      invoid = {invoid}
      """)

#%%
"""
Voy probando rangos de radio para calcular chi
Radio mínimo: tal que en el eje x (xi*Nmean) me de alrededor de 0.1
Radio maximo: tal que la chi no me de inf*
Estos depende del tamaño de la muestra (ngxs)

*tambien sucede que si rmax es muy grande P0 es muy chico, y ln(P0)
crece asintoticamente

rs range for ngxs=10000: np.geomspace(1500,16000,x)
rs range for ngxs=100000: np.geomspace(500,9000,x)
rs range for ngxs=1000000: np.geomspace(200,5000,x)
"""
if ngxs==0: rs = np.geomspace(40,4000,rsbin) 
elif ngxs==10000000: rs = np.geomspace(30,5000,rsbin) 
elif ngxs==1000000: rs = np.geomspace(190,5800,rsbin)
elif ngxs==100000: rs = np.geomspace(800,9100,rsbin)
elif ngxs==10000: rs = np.geomspace(2000,17100,rsbin)
elif ngxs==1000: rs = np.geomspace(7000,27800,rsbin)

rs = np.geomspace(250,4000,10)

gxs = readTNG()
if ngxs!=0:
    np.random.seed(seed)
    ids = np.random.choice(len(gxs),size=ngxs)
    gxs = gxs[ids]

print('Replicating box...')
newgxs = perrep(gxs,lbox,np.max(rs))
print(f'Num of original gxs in box: {len(gxs)}\n\
Num of gxs after replication: {len(newgxs)}')

if zspace == True:
    H0 = .06774
    axis = zspaceAxis
    vaxis = 'v'+axis
    newgxs[axis]+=newgxs[vaxis]/H0
    newgxs[axis][np.where(newgxs[axis]<0.)[0]]+=lbox
    newgxs[axis][np.where(newgxs[axis]>lbox)[0]]-=lbox



pos = np.column_stack((newgxs['x'],newgxs['y'],newgxs['z']))

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

else:
    print('Calculating invoid cic statistics...')
    for i,r in enumerate(rs):
        chi[i], NXi[i], P0[i], N_mean[i], xi_mean[i],\
                    = cic_stats_invoid(tree, nesf, r)

##########
# Writing
##########
if ngxs!=0:
    namefile = f'../data/ngxs{ngxs}_nesf{nesf}'
else:
    namefile = f'../data/allgxs_nesf{nesf}'
if zspace==True: 
    namefile += f'_redshift{axis}'
if invoid == True:
    namefile+= '_invoid'
if jk!=0:
    namefile += '_jk'

namefile += '.npz'
print(f'Creating {namefile}')
if jk!=0:
    np.savez(namefile,chi,chi_std,NXi,NXi_std,P0,P0_std,N_mean,N_mean_std,xi_mean,xi_mean_std,rs)
else:
    np.savez(namefile,chi,NXi,P0,N_mean,xi_mean,rs)


# x = np.geomspace(1E-2,1E3,50)
# c='k'
# chi = -np.log(P0)/N_mean
# NE = N_mean*xi_mean

# plt.plot(x,np.log(1+x)/x,label='Negative Binomial',c=c)
# a=.3
# plt.plot(x,(1/((1-a)*(x/a)))*((1+x/a)**(1-a)-1),label='Generalized Hierarhichal',c=c,ls='--')
# #plt.plot(x,(1-np.e**(-x))/x,label='Minimal')
# plt.plot(x,(np.sqrt(1+2*x)-1)/x,label='Thermodynamical',c=c,ls='-.')
# #plt.plot(x[:-15],1-x[:-15]/2,label='Gauss',c=c)
# # Q=1
# # plt.plot(x,1-(np.euler_gamma+np.log(4*Q*x))/(8*Q),label='BBGKY',c=c,ls=':')

# plt.scatter(NE,chi,lw=2)
# plt.xscale('log')
# plt.legend(loc=3)
# plt.show()
# %%
