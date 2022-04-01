#%%
import numpy as np
from scipy import spatial


def cic_stats(tree, n, r, seed=0):
    """Returns Counts in Cells statistics

    Args:
        tree (ckdtree): coordinates
        n (int): Num of spheres
        r (float): Spheres radius
        nran (int): Num of random points
        seed (int, optional): Random seed. Defaults to 0.

    Returns:
        float: VPF
        float: Mean number of points in spheres of radius r
        float: Averaged 2pcf (variance of counts in cells)
    """
    
    np.random.seed(seed)
    # (b - a) * random_sample() + a
    spheres = (1-2*r)*np.random.rand(n,3)+r
    spheres_tree = spatial.cKDTree(spheres)


    #ngal: Num de gxs en cada esfera de radio r
    ngal = [len(a) for a in spheres_tree.query_ball_tree(tree,r)] 

    #VPF
    P0 = len(np.where(np.array(ngal)==0)[0])/n

    N_mean = np.mean(ngal)

    #xi_mean
    xi_mean = (np.mean((ngal-N_mean)**2)-N_mean)/N_mean**2
    
    return P0, N_mean, xi_mean

import matplotlib.pyplot as plt
import random

nran = 10000
ran_tree = spatial.cKDTree(np.random.rand(nran,3))

rs = np.linspace(.01,.3)
for r in rs:
    P0, N_mean, xi_mean = cic_stats(ran_tree, 1000, r)
    plt.scatter(N_mean*xi_mean,-np.log(P0)/N_mean,color='k')
plt.show()
#%%
ns = np.geomspace(10,10000,30)
for n in ns:
    n=int(n)
    P0, N_mean, xi_mean = cic_stats(ran_tree, n, .3)
    plt.scatter(n,N_mean,color='b')
plt.xscale('log')
plt.show()

# %%
