
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

def cic_stats_jk(tree, n, r, lbox, jkbins):
    """Returns Counts in Cells statistics

    Args:
        tree (ckdtree): coordinates
        n (int): Num of spheres
        r (float): Radius of the spheres
        seed (int, optional): Random seed. Defaults to 0.
        jkbins (int): Num. of divisions per axis for JK resampling

    Returns:
        float: VPF
        float: Mean number of points in spheres of radius r
        float: Averaged 2pcf (variance of counts in cells)
    """
    import numpy as np
    from scipy import spatial
    
    a = r
    b = lbox - r
    n_ = int(n*2) #overhead of centers to then strip off
    # (b - a) * random_sample() + a
    spheres = (b-a)*np.random.rand(n_,3) + a

    rbins = np.linspace(a,b,jkbins+1)
    P0_jk = np.zeros((jkbins,jkbins,jkbins))
    N_mean_jk = np.zeros((jkbins,jkbins,jkbins))
    xi_mean_jk = np.zeros((jkbins,jkbins,jkbins))
    chi_jk = np.zeros((jkbins,jkbins,jkbins))
    NXi_jk = np.zeros((jkbins,jkbins,jkbins))

    for k in range(jkbins):
        mask_z2 = (spheres[:,0] < rbins[k+1])
        mask_z1 = (spheres[:,0] > rbins[k])        
        mask_z = np.logical_and(mask_z1,mask_z2)

        for j in range(jkbins):
            mask_x2 = (spheres[:,0] < rbins[j+1])
            mask_x1 = (spheres[:,0] > rbins[j])
            mask_x = np.logical_and(mask_x1,mask_x2)

            for i in range(jkbins):
                
                mask_y2 = (spheres[:,1] < rbins[i+1])
                mask_y1 = (spheres[:,1] > rbins[i])
                mask_y = np.logical_and(mask_y1,mask_y2)

                mask_xy = np.logical_and(mask_x,mask_y)
                mask_xyz = np.logical_and(mask_xy,mask_z)

                mask = np.invert(mask_xyz)
                sph = spheres[mask,:]
                sph = sph[:n]

                ngal = np.zeros(n)
                for ii in range(n):
                    ngal[ii] = len(tree.query_ball_point(sph[ii],r))


                #VPF
                P0_jk[i,j,k] = len(np.where(ngal==0)[0])/n

                N_mean_jk[i,j,k] = np.mean(ngal)

                #xi_mean
                xi_mean_jk[i,j,k] = (np.mean((ngal-N_mean_jk[i,j,k])**2)-N_mean_jk[i,j,k])/N_mean_jk[i,j,k]**2
                
                chi_jk[i,j,k] = -np.log(P0_jk[i,j,k])/N_mean_jk[i,j,k]
                
                NXi_jk[i,j,k] = N_mean_jk[i,j,k]*xi_mean_jk[i,j,k]
    
    P0 = np.mean(P0_jk.flat)
    P0_std = np.std(P0_jk.flat,ddof=1)

    N_mean = np.mean(N_mean_jk.flat)
    N_mean_std = np.std(N_mean_jk.flat,ddof=1)

    xi_mean = np.mean(xi_mean_jk.flat)
    xi_mean_std = np.std(xi_mean_jk.flat,ddof=1)

    chi = np.mean(chi_jk.flat)
    chi_std = np.std(chi_jk.flat,ddof=1)

    NXi = np.mean(NXi_jk.flat)
    NXi_std = np.std(NXi_jk.flat,ddof=1)

    del ngal, P0_jk, N_mean_jk, xi_mean_jk, chi_jk, NXi_jk
    
    return chi, NXi, P0, N_mean, xi_mean, \
        chi_std, NXi_std, P0_std, N_mean_std, xi_mean_std 