#%%
import numpy as np
import matplotlib.pyplot as plt
from cicTools import *
from scipy import spatial
import configparser
from astropy.io import ascii

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
completeRrange = config['PARAMS'].getboolean('completeRrange')
snap = int(config['PARAMS']['snap']) #snapshot number
minmass = float(config['PARAMS']['minmass']) #log number of minimum mass
minradV = float(config['PARAMS']['minradV']) #minimum void radius

print(f"""
      ngxs = {ngxs}
      nesf = {nesf}
      zspace = {zspace}
      zspaceAxis = {zspaceAxis}
      Num of JK resamplings = {jk}^3
      invoid = {invoid}
      completeRrange = {completeRrange}
      snap = {snap}
      minmass = {minmass}
      minradV = {minradV}
      """)

#
#-----------
# Namefile
#-----------
#

if ngxs!=0:
    namefile = f'../data/dilut{ngxs}_nesf{nesf}'
else:
    namefile = f'../data/allgxs_nesf{nesf}'
if zspace==True: 
    namefile += f'_redshift{zspaceAxis}'
if invoid == True:
    namefile+= '_invoid'
if completeRrange == True:
    namefile+='_allR'
if jk!=0:
    namefile += '_jk'
if snap!=99:
    namefile+=f'_snap{snap}'
if minmass==0.:
    namefile+=f'_minMass1e10'
elif minmass==1.:
    namefile+=f'_minMass1e11'
namefile += f'_minradV{minradV}'

if minmass==-1.: voidsfile='../data/tng300-1_voids.dat'
elif minmass==0.: voidsfile='../data/voids_1e10.dat'
elif minmass==1.: voidsfile='../data/voids_1e11.dat'
print('void file location:', voidsfile)


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
# if ngxs==0: rs = np.geomspace(40,4000,rsbin) 
# elif ngxs==10000000: rs = np.geomspace(30,5000,rsbin) 
# elif ngxs==1000000: rs = np.geomspace(190,5800,rsbin)
# elif ngxs==100000: rs = np.geomspace(800,9100,rsbin)
# elif ngxs==10000: rs = np.geomspace(2000,17100,rsbin)
# elif ngxs==1000: rs = np.geomspace(7000,27800,rsbin)

# if invoid==True:
#     if ngxs==0: rs = np.geomspace(250,2800,10) 
#     elif ngxs==10000000: rs = np.geomspace(300,3000,10) 
#     elif ngxs==1000000: rs = np.geomspace(700,3500,10)

if completeRrange==False: 
    rs = np.geomspace(250,2500,rsbin) #Dejo esto para que tome algún valor en caso que invoid==False
    if invoid==True:
        if minradV==7.:
            rs = np.geomspace(250,2500,rsbin) 
        elif minradV==9.:
            rs = np.geomspace(1500,6500,rsbin)

if completeRrange==True: rs = np.geomspace(40,4000,rsbin)


gxs = readTNG(snap=snap,minmass=minmass)
if ngxs!=0:
    np.random.seed(seed)
    ids = np.random.choice(len(gxs),size=int(len(gxs)*ngxs))
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

if invoid == True:

    voids = ascii.read(voidsfile,\
        names=['r','x','y','z','vx','vy','vz',\
            'deltaint_1r','maxdeltaint_2-3r','log10Poisson','Nrecenter'])
    voids = voids[voids['r']>=minradV]
    print('N of voids:',len(voids))
    voids['r'] = voids['r']*1000 #Converts kpc to Mpc
    voids['x'] = voids['x']*1000
    voids['y'] = voids['y']*1000
    voids['z'] = voids['z']*1000

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
                        = cic_stats_invoid_jk(voids, tree, nesf, r)

    else:
        print('Calculating invoid cic statistics...')
        for i,r in enumerate(rs):
            chi[i], NXi[i], P0[i], N_mean[i], xi_mean[i],\
                        = cic_stats_invoid(voids, tree, nesf, r)

##########
# Writing
##########

namefile += '.npz'
print(f'Creating {namefile}')
if jk!=0:
    np.savez(namefile,chi,chi_std,NXi,NXi_std,P0,P0_std,N_mean,N_mean_std,xi_mean,xi_mean_std,rs)
else:
    np.savez(namefile,chi,NXi,P0,N_mean,xi_mean,rs)

print(chi, rs)

#%%
# x = np.geomspace(1E-2,1E3,50)
# c='k'
# #chi = -np.log(P0)/N_mean
# #NE = N_mean*xi_mean

# plt.plot(x,np.log(1+x)/x,label='Negative Binomial',c=c)
# #a=.3
# #plt.plot(x,(1/((1-a)*(x/a)))*((1+x/a)**(1-a)-1),label='Generalized Hierarhichal',c=c,ls='--')
# #plt.plot(x,(1-np.e**(-x))/x,label='Minimal')
# plt.plot(x,(np.sqrt(1+2*x)-1)/x,label='Thermodynamical',c=c,ls='-.')
# #plt.plot(x[:-15],1-x[:-15]/2,label='Gauss',c=c)
# # Q=1
# # plt.plot(x,1-(np.euler_gamma+np.log(4*Q*x))/(8*Q),label='BBGKY',c=c,ls=':')

# plt.plot(NXi,chi,lw=2)
# plt.xscale('log')
# plt.legend(loc=3)
# plt.show()
# %%
