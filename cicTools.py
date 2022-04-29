
def cic_stats(tree, n, r, lbox):
    """Returns Counts in Cells statistics

    Args:
        tree (ckdtree): coordinates
        n (int): Num of spheres
        r (float): Radius of the spheres
        seed (int, optional): Random seed. Defaults to 0.

    Returns:
        float: VPF
        float: Mean number of points in spheres of radius r
        float: Averaged 2pcf (variance of counts in cells)
    """
    import numpy as np
    from scipy import spatial
    
    a = r
    b = lbox - r    
    # (b - a) * random_sample() + a
    spheres = (b-a)*np.random.rand(n,3) + a
    #spheres_tree = spatial.cKDTree(spheres)


    #ngal: Num de gxs en cada esfera de radio r
    #ngal = [len(a) for a in spheres_tree.query_ball_tree(tree,r)] 

    #Otra forma de obtener ngal:
    ngal = np.zeros(n)
    for k in range(n):
        ngal[k] = len(tree.query_ball_point(spheres[k],r))


    #VPF
    P0 = len(np.where(ngal==0)[0])/n

    N_mean = np.mean(ngal)

    #xi_mean
    xi_mean = (np.mean((ngal-N_mean)**2)-N_mean)/N_mean**2
    
    del ngal
    
    return P0, N_mean, xi_mean


def readTNG():
    """
    Read subhalos/galaxies in the TNG300-1 simulation 

    gxs: an ascii Table with the fields and filters I usually need for this: Position, Mass, Spin

    """
    import sys
    illustrisPath = '/home/fede'
    basePath = '../../../TNG300-1/output/'
    sys.path.append(illustrisPath)
    import illustris_python as il
    import numpy as np
    from astropy.table import Table
    import random 
    
    mass = il.groupcat.loadSubhalos(basePath,99,fields=['SubhaloMass'])                                                                                                                      
    ids = np.where((np.log10(mass)>-1.)&(np.log10(mass)<3.))
    mass = mass[ids]

    pos = il.groupcat.loadSubhalos(basePath,99,fields=['SubhaloPos'])
    pos = pos[ids]

    vel = il.groupcat.loadSubhalos(basePath,99,fields=['SubhaloVel'])
    vel = vel[ids]


    #gxs = Table(np.column_stack([pos[:,0],pos[:,1],pos[:,2],mass]),names=['x','y','z','mass'])    
    gxs = Table(np.column_stack([pos[:,0],pos[:,1],pos[:,2],vel[:,0],vel[:,1],vel[:,2]]),names=['x','y','z','vx','vy','vz'])    
    
    del mass,pos,vel

    return gxs