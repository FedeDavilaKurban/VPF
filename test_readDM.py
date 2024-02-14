"""
Script de prueba

Quiero leer DM de illustris y plottearla en un hist2D

El problema es que es muy densa.

La solución que estoy probando es leer, diluir, y guardar. Finalmente plottear
"""
#%%
import sys
illustrisPath = '/home/fdavilakurban/'
#basePath = '../../../TNG300-1/output/'
basePath='/media/fdavilakurban/0a842929-67de-4adc-b64c-8bc6d17a08b0/fdavilakurban/TNG300-1/output'
sys.path.append(illustrisPath)
import illustris_python as il
import numpy as np
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import time

def pSplitRange(indrange, numProcs, curProc, inclusive=False):
    """ Divide work for embarassingly parallel problems. 
    Accept a 2-tuple of [start,end] indices and return a new range subset.
    If inclusive==True, then assume the range subset will be used e.g. as input to snapshotSubseet(),
    which unlike numpy convention is inclusive in the indices."""
    assert len(indrange) == 2 and indrange[1] > indrange[0]

    if numProcs == 1:
        if curProc != 0:
            raise Exception("Only a single processor but requested curProc>0.")
        return indrange

    # split array into numProcs segments, and return the curProc'th segment
    splitSize = int(np.floor( (indrange[1]-indrange[0]) / numProcs ))
    start = indrange[0] + curProc*splitSize
    end   = indrange[0] + (curProc+1)*splitSize

    # for last split, make sure it takes any leftovers
    if curProc == numProcs-1:
        end = indrange[1]

    if inclusive and curProc < numProcs-1:
        # not for last split/final index, because this should be e.g. NumPart[0]-1 already
        end -= 1

    return [start,end]


def loadSubset(simPath, snap, partType, fields, chunkNum=0, totNumChunks=1):
    """ Load part of a snapshot. """
    nTypes = 6
    ptNum = il.util.partTypeNum(partType)

    with h5py.File(il.snapshot.snapPath(simPath,snap),'r') as f:
        numPartTot = il.snapshot.getNumPart( dict(f['Header'].attrs.items()) )[ptNum]

    # define index range
    indRange_fullSnap = [0,numPartTot-1]
    indRange = pSplitRange(indRange_fullSnap, totNumChunks, chunkNum, inclusive=True)

    # load a contiguous chunk by making a subset specification in analogy to the group ordered loads
    subset = { 'offsetType'  : np.zeros(nTypes, dtype='int64'),
               'lenType'     : np.zeros(nTypes, dtype='int64') }

    subset['offsetType'][ptNum] = indRange[0]
    subset['lenType'][ptNum]    = indRange[1]-indRange[0]+1

    # add snap offsets (as required)
    with h5py.File(il.snapshot.offsetPath(simPath,snap),'r') as f:
        subset['snapOffsets'] = np.transpose(f['FileOffsets/SnapByType'][()])

    # load from disk
    r = il.snapshot.loadSubset(simPath, snap, partType, fields, subset=subset)

    return r


#%%
nSubLoads = 500
dilution = .5 #percent

dm_pos = loadSubset(basePath,99,'dm',['Coordinates'],chunkNum=0,totNumChunks=nSubLoads)
dm_vel = loadSubset(basePath,99,'dm',['Velocities'],chunkNum=0,totNumChunks=nSubLoads)
randomID = np.random.randint(len(dm_pos),size=int(len(dm_pos)*dilution/100))
pos_diluted = dm_pos[randomID]
vel_diluted = dm_vel[randomID]

print(pos_diluted)
print(len(pos_diluted))
print(vel_diluted)
print(len(vel_diluted))

for i in range(1,nSubLoads):
    #t1=time.time()
    dm_pos = loadSubset(basePath,99,'dm',['Coordinates'],chunkNum=i,totNumChunks=nSubLoads)
    dm_vel = loadSubset(basePath,99,'dm',['Velocities'],chunkNum=i,totNumChunks=nSubLoads)

    randomID = np.random.randint(len(dm_pos),size=int(len(dm_pos)*dilution/100))
    
    pos_diluted = np.append(pos_diluted,dm_pos[randomID],axis=0)
    vel_diluted = np.append(vel_diluted,dm_vel[randomID],axis=0)
    #print(i,time.time()-t1)
    print(i)
    #print(len(pos_diluted))


np.savez('../data/dmpos_diluted.npz',pos_diluted,vel_diluted)
#%%

fig, ax = plt.subplots()

ax.hist2d(pos_diluted[:,0], pos_diluted[:,1], norm=mpl.colors.LogNorm(), bins=32)

ax.set_xlim([0,205000])
ax.set_ylim([0,205000])
ax.set_xlabel('x [ckpc/h]')
ax.set_ylabel('y [ckpc/h]')

plt.show()