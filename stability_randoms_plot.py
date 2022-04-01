#%%
import numpy as np
import matplotlib.pyplot as plt
from cicTools import cic_stats
from scipy import spatial

r = .04
nran = 10000
ran_tree = spatial.cKDTree(np.random.rand(nran,3))
print('N_tot=',nran)
print('Mean interparticle distance:',nran**(-1/3))
print('Wigner-Seitz radius:',(3/(4*np.pi*nran))**(1/3))
print('Testing radius:',r)

ns = np.geomspace(10,1000000,30)

P0 = np.zeros(len(ns))
N_mean = np.zeros(len(ns))
xi_mean = np.zeros(len(ns))

for i,n in enumerate(ns):
    n=int(n)
    P0[i], N_mean[i], xi_mean[i] = cic_stats(ran_tree, n, r, seed=0)

    
fig= plt.figure(figsize=(6,9))
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)
ax1.get_shared_x_axes().join(ax1, ax2, ax3)
plt.rcParams['font.size'] = 14

"""
Nmean
"""
ax1.scatter(ns,N_mean)

ax1.text(.7,.9,r'$N_{Tot}=$'+f'{nran}', transform=ax1.transAxes)
ax1.text(.7,.8,r'$R=$'+f'{r}', transform=ax1.transAxes)

ax1.set_ylabel(r'$\bar{N}(R)$')
ax1.set_xscale('log')

"""
P0
"""
ax2.scatter(ns,P0)

ax2.set_ylabel(r'$P_0(R)$')
ax2.set_xscale('log')

"""
Xi_mean
"""
ax3.scatter(ns,xi_mean)

ax3.set_xlabel('Num. of spheres')
ax3.set_ylabel(r'$\bar{\xi}(R)$')
ax3.set_xscale('log')

ax1.set_xticklabels([])
ax2.set_xticklabels([])

plt.tight_layout()
plt.savefig('../plots/stability_randoms.png')
plt.show()
# %%
