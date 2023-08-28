#%%
import numpy as np
import matplotlib.pyplot as plt
from cicTools import cic_stats, cic_stats_jk
from scipy import spatial
#%%
r = 800
lbox = 205000
nran = 13333606
seed = 818237
np.random.seed(seed)
ranpos = float(lbox)*np.random.rand(nran,3)
ran_tree = spatial.cKDTree(ranpos)
print('N_tot = ',nran)
print('Mean interparticle distance:',nran**(-1/3))
print('Wigner-Seitz radius:',(3/(4*np.pi*nran))**(1/3))
print('Testing radius:',r)

ns = np.geomspace(100,1000000,10).astype(int)

P0 = np.zeros(len(ns))
N_mean = np.zeros(len(ns))
xi_mean = np.zeros(len(ns))
chi = np.zeros(len(ns))
NXi = np.zeros(len(ns))
chi_std = np.zeros(len(ns))
NXi_std = np.zeros(len(ns))
P0_std = np.zeros(len(ns))
N_mean_std = np.zeros(len(ns))
xi_mean_std = np.zeros(len(ns))

for i,n in enumerate(ns):
    #chi[i], NXi[i], P0[i], N_mean[i], xi_mean[i] = cic_stats(ran_tree, n, r, lbox)
    chi[i], NXi[i], P0[i], N_mean[i], xi_mean[i], \
            chi_std[i], NXi_std[i], P0_std[i], N_mean_std[i], xi_mean_std[i] = cic_stats_jk(ran_tree, n, r, lbox, jkbins=3)

#%%
#namefile = '../data/stability_randoms.npz'
#np.savez(namefile,chi,NXi,P0,N_mean,xi_mean,ns)
#%%
namefile = '../data/stability_randoms_jk.npz'

np.savez(namefile,chi, NXi, P0, N_mean, xi_mean, \
    chi_std, NXi_std, P0_std, N_mean_std,xi_mean_std)
#%%

fig= plt.figure(figsize=(6,9))
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)
ax1.get_shared_x_axes().join(ax1, ax2, ax3)
plt.rcParams['font.size'] = 14


"""
Nmean
"""
N_mean_analytical = len(ranpos)*(4*np.pi*r**3/3)/lbox**3
ax1.hlines(N_mean_analytical,np.min(ns),np.max(ns),ls=':')
ax1.plot(ns,N_mean)

ax1.text(.7,.9,r'$N_{Tot}=$'+f'{nran}', transform=ax1.transAxes)
ax1.text(.7,.8,r'$R=$'+f'{r}', transform=ax1.transAxes)

ax1.set_ylabel(r'$\bar{N}(R)$')
ax1.set_xscale('log')

"""
P0
"""
ax2.plot(ns,P0)

ax2.set_ylabel(r'$P_0(R)$')
ax2.set_xscale('log')

"""
Xi_mean
"""
ax3.plot(ns,xi_mean)

ax3.set_xlabel('Num. of spheres')
ax3.set_ylabel(r'$\bar{\xi}(R)$')
ax3.set_xscale('log')

ax1.set_xticklabels([])
ax2.set_xticklabels([])

plt.tight_layout()
plt.savefig('../plots/stability_randoms.png')
plt.show()
# %%

#%%
#
#Estos son los plots con JK
#
import matplotlib.pyplot as plt

from cicTools import delta_P0

r = .004
lbox = 1
nran = 10000000

namefile = f'../data/stability_randoms_jk.npz'
stats = np.load(namefile)

chi = stats['arr_0']
NXi = stats['arr_1']
P0 = stats['arr_2']
N_mean = stats['arr_3']
xi_mean = stats['arr_4']
chi_std = stats['arr_5']
NXi_std = stats['arr_6']
P0_std = stats['arr_7']
N_mean_std = stats['arr_8']
xi_mean_std = stats['arr_9']
#ns = np.geomspace(10,1000000,30).astype(int)

P0err = delta_P0(P0,ns)


fig= plt.figure(figsize=(6,9))
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)
ax1.get_shared_x_axes().join(ax1, ax2, ax3)
plt.rcParams['font.size'] = 14

"""
Nmean
"""
N_mean_analytical = nran*(4*np.pi*r**3/3)/(lbox)**3
ax1.hlines(N_mean_analytical,np.min(ns),np.max(ns),ls=':',color='k')
ax1.errorbar(ns,N_mean,yerr=N_mean_std,marker='o',capsize=3)
#ax1.plot(ns,N_mean)

ax1.text(.6,.9,r'$n_\mathrm{ran}=$'+f'{nran}', transform=ax1.transAxes)
ax1.text(.6,.7,r'$R=$'+f'{r}', transform=ax1.transAxes)
ax1.text(.6,.8,r'$L_\mathrm{box}=1$', transform=ax1.transAxes)

ax1.set_ylabel(r'$\bar{N}$')
ax1.set_xscale('log')

"""
P0
"""
P0_ran = np.exp(-(nran/lbox**3)*(4./3.)*np.pi*r**3) #theoretical value of P0 for poisson dist.
ax2.errorbar(ns,P0,yerr=P0err,marker='o',capsize=3)
ax2.hlines(P0_ran,np.min(ns),np.max(ns),ls=':',color='k')

ax2.set_ylabel(r'$P_0$')
ax2.set_xscale('log')

"""
Xi_mean
"""
ax3.errorbar(ns,xi_mean,yerr=xi_mean_std,marker='o',capsize=3)

ax3.hlines(0,np.min(ns),np.max(ns),ls=':',color='k')

ax3.set_xlabel(r'$N_\mathrm{esf}$')
ax3.set_ylabel(r'$\bar{\xi}$')
ax3.set_xscale('log')

ax1.set_xticklabels([])
ax2.set_xticklabels([])

ax1.set_title('Random sample')
plt.tight_layout()
#plt.savefig('../plots/stability_randoms_jk.png')
plt.show()
# %%
