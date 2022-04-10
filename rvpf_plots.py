#%%
import numpy as np
import matplotlib.pyplot as plt
#%%

lbox = 205000
ntot = 100000
n = 100000

namefile = f'../data/ngxs{ntot}_nesf{n}.npz'
stats = np.load(namefile)

P0 = stats['arr_0']
N_mean = stats['arr_1']
xi_mean = stats['arr_2']
rs = stats['arr_3']

chi = -np.log(P0)/N_mean
NE = N_mean*xi_mean

x = np.geomspace(1E-1,1E1,50)
c='k'

plt.plot(x,np.log(1+x)/x,label='Negative Binomial',c=c)
a=.3
plt.plot(x,(1/((1-a)*(x/a)))*((1+x/a)**(1-a)-1),label='Generalized Hierarhichal',c=c,ls='--')
#plt.plot(x,(1-np.e**(-x))/x,label='Minimal')
plt.plot(x,(np.sqrt(1+2*x)-1)/x,label='Thermodynamical',c=c,ls='-.')
#plt.plot(x,1-x/2,label='Gauss')
Q=1
#plt.plot(x,1-(np.euler_gamma+np.log(4*Q*x))/(8*Q),label='BBGKY',c=c,ls=':')

plt.scatter(NE,chi,lw=3)


ntot = 10000

namefile = f'../data/ngxs{ntot}_nesf{n}.npz'
stats = np.load(namefile)

P0 = stats['arr_0']
N_mean = stats['arr_1']
xi_mean = stats['arr_2']
rs = stats['arr_3']

chi = -np.log(P0)/N_mean
NE = N_mean*xi_mean

plt.scatter(NE,chi,lw=3)

plt.xscale('log')
plt.legend(loc=3)
plt.show()

#%%
x = np.geomspace(1E-2,1E2,50)
c='k'
plt.plot(x,np.log(1+x)/x,label='Negative Binomial',c=c)
a=.3
plt.plot(x,(1/((1-a)*(x/a)))*((1+x/a)**(1-a)-1),label='Generalized Hierarhichal',c=c,ls='--')
plt.plot(x,(np.sqrt(1+2*x)-1)/x,label='Thermodynamical',c=c,ls='-.')

for ngxs in [10000,100000,1000000]:
    
    namefile = f'../data/ngxs{ngxs}_nesf{n}.npz'
    stats = np.load(namefile)

    P0 = stats['arr_0']
    N_mean = stats['arr_1']
    xi_mean = stats['arr_2']
    rs = stats['arr_3']

    chi = -np.log(P0)/N_mean
    NE = N_mean*xi_mean

    plt.plot(NE,chi,ls='-',label=n)

plt.xscale('log')   
plt.legend(loc=3)
plt.show()
#%%
# """
# RANDOMS
# """

# chi_ran = -np.log(P0_ran)/Nmean_ran
# NE_ran = Nmean_ran*ximean_ran
# #%%
# x = np.geomspace(1E-3,1E3,50)
# plt.plot(x,np.log(1+x)/x)
# plt.scatter(NE_ran,chi_ran)
# plt.xscale('log')
# #plt.yscale('log')
# plt.show()
# # %%
