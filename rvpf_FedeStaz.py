"""
El Fede Staz me pidio que vea como varia la VPF con la masa barionica

Copié el programa rvpf_jk.py

La diferencia es que copio la función readTNG del cicTools con el motivo de 
leer las masas y restarle la DM (SubhaloMassTypes[1])

La funcion la renombro readTNG_ y la copio abajo
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from cicTools import *
from scipy import spatial
import configparser
from astropy.io import ascii

def readTNG_(snap=99,minmass=-1.,maxmass=3.):
    """
    Read subhalos/galaxies in the TNG300-1 simulation 

    Args:
        disk (string): where to read TNG. 'local' or 'mounted'
        snap (int): snapshot number
        minmass, maxmass (float): log10 of min/max mass thresholds (e.g.: -1 means 1E-9Mdot, 3 means 1E13Mdot)
            
    Returns:
        ascii Table: gxs (Position, Mass, Velocity)

    """
    import sys
    illustrisPath = '/home/fdavilakurban/'
    basePath = '../../../TNG300-1/output/'

    sys.path.append(illustrisPath)
    import illustris_python as il
    import numpy as np
    from astropy.table import Table
    import random 
    
    mass = il.groupcat.loadSubhalos(basePath,snap,fields=['SubhaloMass'])
    masses = il.groupcat.loadSubhalos(basePath,snap,fields=['SubhaloMassType'])                                                                                                                      
    dm = masses[:,1]
    bmass = mass-dm 
    #bmass_alt=np.sum(np.array([masses[:,0],masses[:,4],masses[0:,5]]),axis=0)
    bmass = bmass[np.where(bmass!=0.)[0]]

    ids = np.where((np.log10(mass)>minmass)&(np.log10(mass)<maxmass))

    pos = il.groupcat.loadSubhalos(basePath,snap,fields=['SubhaloPos'])
    pos = pos[ids]

    vel = il.groupcat.loadSubhalos(basePath,snap,fields=['SubhaloVel'])
    vel = vel[ids]

    gxs = Table(np.column_stack([pos[:,0],pos[:,1],pos[:,2],vel[:,0],vel[:,1],vel[:,2]]),names=['x','y','z','vx','vy','vz'])    
    
    del mass,masses,pos,vel,bmass,dm

    return gxs

config = configparser.ConfigParser()
config.read('config.ini')

write = config['PARAMS'].getboolean('write') #write files with results
plot = config['PARAMS'].getboolean('plot') #plot results for checking 

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
minmass = float(config['PARAMS']['minmass']) #log of minimum mass
maxmass = float(config['PARAMS']['maxmass']) #log of minimum mass
minradV = float(config['PARAMS']['minradV']) #minimum void radius
voidfile = str(config['PARAMS']['voidfile']) #location of voids file / which voids to use
delta = str(config['PARAMS']['delta']) #delta used in void identification
voids_zs = config['PARAMS'].getboolean('voids_zs') #read voids identified in z-space
evolDelta = config['PARAMS'].getboolean('evolDelta') #read voids identified with evolved integrated delta

if invoid==True:
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
        maxmass = {maxmass}
        minradV = {minradV}
        evolDelta = {evolDelta}
        voidfile = {voidfile}
        """)
elif invoid==False:
    print(f"""
        ngxs = {ngxs}
        nesf = {nesf}
        zspace = {zspace}
        zspaceAxis = {zspaceAxis}
        Num of JK resamplings = {jk}^3
        completeRrange = {completeRrange}
        snap = {snap}
        minmass = {minmass}
        maxmass = {maxmass}
        """)

#
#-----------
# Print voids name file
#-----------
#
if invoid==True:
    if zspace==False:
        if evolDelta==False:
            if delta=='09':
                if voidfile=='1e9': voidsfile='../data/tng300-1_voids.dat'
                elif voidfile=='1e10': voidsfile='../data/voids_1e10.dat'
                elif voidfile=='1e11': voidsfile='../data/voids_1e11.dat'
            if delta=='08':
                if voidfile=='1e9': voidsfile='../data/voids_1e9_08.dat'
                elif voidfile=='1e10': voidsfile='../data/voids_1e10_08.dat'
                elif voidfile=='1e11': voidsfile='../data/voids_1e11_08.dat'
            if delta=='07':
                if voidfile=='1e9': voidsfile='../data/voids_1e11_07.dat'
                elif voidfile=='1e10': voidsfile='../data/voids_1e10_07.dat'
                elif voidfile=='1e11': voidsfile='../data/voids_1e11_07.dat'
        if evolDelta==True:
            if voidfile!='1e11': raise Exception('Voids not identified with evolved delta for this "voidfile" value')
            voidsfile = f'../data/voids_1e11_snap{snap}.dat'

    elif zspace==True:
        if evolDelta==False:
            if delta=='09':
                if voidfile=='1e9': voidsfile='../data/voids_zs_1e9_09.dat'
                elif voidfile=='1e10': voidsfile='../data/voids_zs_1e10_09.dat'
                elif voidfile=='1e11': voidsfile='../data/voids_zs_1e11_09.dat'
            if delta=='08':
                if voidfile=='1e9': voidsfile='../data/voids_zs_1e9_08.dat'
                elif voidfile=='1e10': voidsfile='../data/voids_zs_1e10_08.dat'
                elif voidfile=='1e11': voidsfile='../data/voids_zs_1e11_08.dat'
            if delta=='07':
                if voidfile=='1e9': voidsfile='../data/voids_zs_1e9_09.dat'
                elif voidfile=='1e10': voidsfile='../data/voids_zs_1e10_07.dat'
                elif voidfile=='1e11': voidsfile='../data/voids_zs_1e11_07.dat'
        if evolDelta==True:
            if voidfile!='1e11': raise Exception('Voids not identified with evolved delta for this "voidfile" value')
            voidsfile = f'../data/voids_zs_1e11_snap{snap}.dat'
    print('void file location:', voidsfile)

#
#-----------
# Namefile
#-----------
#
if write==True:
    if ngxs!=0:
        namefile = f'../data/paraFedeStasz/dilut{ngxs}_nesf{nesf}'
    else:
        namefile = f'../data/paraFedeStasz/allgxs_nesf{nesf}'
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
    elif minmass==2.:
        namefile+=f'_minMass1e12'
    elif minmass not in [-1.,0.,1.,2.]:
        raise Exception('Invalid minmass value')
    if invoid==True:
        if evolDelta==False:
            if voidfile=='1e9': namefile+='_v1e9'
            elif voidfile=='1e10': namefile+='_v1e10'
            elif voidfile=='1e11': namefile+='_v1e11'
            namefile += f'_minradV{minradV}'
        if evolDelta==True:
            if voidfile=='1e11': 
                namefile+='_v1e11EvolDelta'
            else: 
                raise Exception('Voids not identified with evolved delta for this "voidfile" value')
            namefile += f'_minradV{minradV}'
    if delta!='09':
        namefile += f'_d{delta}'
    namefile += '.npz'

    print('Filename to be created:',namefile)




#%%

if invoid==True:

    try:
        voids = ascii.read(voidsfile,\
            names=['r','x','y','z','vx','vy','vz',\
                'deltaint_1r','maxdeltaint_2-3r','log10Poisson','Nrecenter'])
    except:
        voids = ascii.read(voidsfile,\
            names=['r','x','y','z','vx','vy','vz',\
                'deltaint_1r','maxdeltaint_2-3r'])


    voids = voids[voids['r']>=minradV]
    print('N of voids:',len(voids))
    voids['r'] = voids['r']*1000 #Converts kpc to Mpc
    voids['x'] = voids['x']*1000
    voids['y'] = voids['y']*1000
    voids['z'] = voids['z']*1000

    maxrad = round(np.min(voids['r']))
    rs = np.geomspace(maxrad/40.,maxrad/4.,rsbin)
else:
    rs = np.geomspace(40,4000,rsbin)


#
#-----------
# Read data from Illustris
#-----------
#
gxs = readTNG_(snap=snap,minmass=minmass,maxmass=maxmass)
if ngxs!=0:
    np.random.seed(seed)
    ids = np.random.choice(len(gxs),size=int(len(gxs)*ngxs))
    gxs = gxs[ids]

#
#-----------
# Replicate box edges periodically
#-----------
#
print('Replicating box:')
newgxs = perrep(gxs,lbox,np.max(rs))
print(f'Num of original gxs in box: {len(gxs)}\n\
Num of gxs after replication: {len(newgxs)}')

#
#-----------
# Distant observer aproximation for z-space
#-----------
#
if zspace == True:
    H0 = .06774
    axis = zspaceAxis
    vaxis = 'v'+axis
    newgxs[axis]+=newgxs[vaxis]/H0
    newgxs[axis][np.where(newgxs[axis]<0.)[0]]+=lbox
    newgxs[axis][np.where(newgxs[axis]>lbox)[0]]-=lbox


#
#-----------
# VPF calculations
#-----------
#
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


#
#-----------
# Writing file
#-----------
#
if write==True:
    print(f'Creating {namefile}')
    if jk!=0:
        np.savez(namefile,chi,chi_std,NXi,NXi_std,P0,P0_std,N_mean,N_mean_std,xi_mean,xi_mean_std,rs)
    else:
        np.savez(namefile,chi,NXi,P0,N_mean,xi_mean,rs)

print(chi, rs)

#%%
if plot==True:
    x = np.geomspace(1E-2,1E3,50)
    c='k'

    plt.plot(x,np.log(1+x)/x,label='Negative Binomial',c=c)
    plt.plot(x,(np.sqrt(1+2*x)-1)/x,label='Thermodynamical',c=c,ls='-.')

    plt.plot(NXi,chi,lw=2)
    plt.xscale('log')
    plt.legend(loc=3)
    plt.show()
# %%
