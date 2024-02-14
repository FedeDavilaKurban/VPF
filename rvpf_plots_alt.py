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

plt.scatter(NE,chi,lw=3,label=f'ngxs={ntot}')


ntot = 10000

namefile = f'../data/ngxs{ntot}_nesf{n}.npz'
stats = np.load(namefile)

P0 = stats['arr_0']
N_mean = stats['arr_1']
xi_mean = stats['arr_2']
rs = stats['arr_3']

chi = -np.log(P0)/N_mean
NE = N_mean*xi_mean

plt.scatter(NE,chi,lw=3,label=f'ngxs={ntot}')

plt.xscale('log')
plt.legend(loc=3)
plt.show()

#%%
"""
Comparacion entre distintos ngxs en realspace
"""
plt.figure(figsize=(9,6))
plt.rcParams['font.size'] = 14
x = np.geomspace(1E-2,1E2,50)
n = 100000
c='k'
plt.plot(x,np.log(1+x)/x,label='Negative Binomial',c=c)
a=.3
plt.plot(x,(1/((1-a)*(x/a)))*((1+x/a)**(1-a)-1),label='Generalized Hierarhichal',c=c,ls='--')
plt.plot(x,(np.sqrt(1+2*x)-1)/x,label='Thermodynamical',c=c,ls='-.')

for ngxs in [1000,10000,100000,1000000,10000000]:
    
    namefile = f'../data/ngxs{ngxs}_nesf{n}.npz'
    stats = np.load(namefile)

    P0 = stats['arr_0']
    N_mean = stats['arr_1']
    xi_mean = stats['arr_2']
    rs = stats['arr_3']

    chi = -np.log(P0)/N_mean
    NE = N_mean*xi_mean

    plt.scatter(NE,chi,ls='-',label=ngxs)

plt.xscale('log')   
plt.legend(loc=3)
plt.show()
#%%
"""
Comparacion entre distintos ngxs en zspace
"""
plt.figure(figsize=(9,6))
plt.rcParams['font.size'] = 14
x = np.geomspace(1E-2,1E2,50)
n = 100000
c='k'
plt.plot(x,np.log(1+x)/(x),label='Negative Binomial',c=c)
a=.3
plt.plot(x,(1/((1-a)*(x/a)))*((1+x/a)**(1-a)-1),label='Generalized Hierarhichal',c=c,ls='--')
plt.plot(x,(np.sqrt(1+2*x)-1)/x,label='Thermodynamical',c=c,ls='-.')

for ngxs in [1000,10000,100000,1000000,10000000]:
    
    namefile = f'../data/ngxs{ngxs}_nesf{n}_redshiftx.npz'
    stats = np.load(namefile)

    P0 = stats['arr_0']
    N_mean = stats['arr_1']
    xi_mean = stats['arr_2']
    rs = stats['arr_3']

    chi = -np.log(P0)/N_mean
    NE = N_mean*xi_mean

    plt.scatter(NE,chi,ls='-',label=ngxs)

plt.xscale('log')   
plt.legend(loc=3)
plt.show()
#%%
"""
comparacion para el redshifteo de los 3 ejes
"""
plt.figure(figsize=(9,6))
plt.rcParams['font.size'] = 14
x = np.geomspace(1E-2,1E2,50)
n = 100000
c='k'
plt.plot(x,np.log(1+x)/x,label='Negative Binomial',c=c)
a=.3
plt.plot(x,(1/((1-a)*(x/a)))*((1+x/a)**(1-a)-1),label='Generalized Hierarhichal',c=c,ls='--')
plt.plot(x,(np.sqrt(1+2*x)-1)/x,label='Thermodynamical',c=c,ls='-.')

ngxs = 1000000

for axis in ['x','y','z']:
    
    namefile = f'../data/ngxs{ngxs}_nesf{n}_redshift{axis}.npz'
    stats = np.load(namefile)

    P0 = stats['arr_0']
    N_mean = stats['arr_1']
    xi_mean = stats['arr_2']
    rs = stats['arr_3']

    chi = -np.log(P0)/N_mean
    NE = N_mean*xi_mean

    plt.plot(NE,chi,ls='-',label=axis)

plt.xscale('log')   
plt.legend(loc=3)
plt.show()

#%%
"""
Comparacion entre real space y z space
"""
plt.figure(figsize=(9,6))
plt.rcParams['font.size'] = 14

x = np.geomspace(1E-2,1E2,50)
n = 100000
c='k'
plt.plot(x,np.log(1+x)/x,label='Negative Binomial',c=c)
a=.3
plt.plot(x,(1/((1-a)*(x/a)))*((1+x/a)**(1-a)-1),label='Generalized Hierarhichal',c=c,ls='--')
plt.plot(x,(np.sqrt(1+2*x)-1)/x,label='Thermodynamical',c=c,ls='-.')

ngxs = 1000000

#Z SPACE
namefile = f'../data/ngxs{ngxs}_nesf{n}_redshiftx.npz'
stats = np.load(namefile)
P0 = stats['arr_0']
N_mean = stats['arr_1']
xi_mean = stats['arr_2']
rs = stats['arr_3']
chi = -np.log(P0)/N_mean
NE = N_mean*xi_mean
plt.plot(NE,chi,lw=3,ls='-',label='zspace')

#REAL SPACE
namefile = f'../data/ngxs{ngxs}_nesf{n}.npz'
stats = np.load(namefile)
P0 = stats['arr_0']
N_mean = stats['arr_1']
xi_mean = stats['arr_2']
rs = stats['arr_3']
chi = -np.log(P0)/N_mean
NE = N_mean*xi_mean
plt.plot(NE,chi,lw=3,ls='-',label='realspace')

plt.xscale('log')   
plt.legend(loc=3)
plt.show()
#%%
"""
Comparacion entre distintos ngxs y realspace/zspace
"""
plt.figure(figsize=(9,6))
plt.rcParams['font.size'] = 14

x = np.geomspace(1E-1,1E2,50)
n = 100000
c='k'
plt.plot(x,np.log(1+x)/x,label='Negative Binomial',c=c)
a=.3
plt.plot(x,(1/((1-a)*(x/a)))*((1+x/a)**(1-a)-1),label='Generalized Hierarhichal',c=c,ls='--')
plt.plot(x,(np.sqrt(1+2*x)-1)/x,label='Thermodynamical',c=c,ls='-.')

ngxss = [1000,10000,100000]

c='b'
for ngxs in ngxss:
    
    namefile = f'../data/ngxs{ngxs}_nesf{n}.npz'
    stats = np.load(namefile)

    P0 = stats['arr_0']
    N_mean = stats['arr_1']
    xi_mean = stats['arr_2']
    rs = stats['arr_3']

    chi = -np.log(P0)/N_mean
    NE = N_mean*xi_mean

    plt.plot(NE,chi,ls='-',c=c)#,label=ngxs)
    
c='r'
for ngxs in ngxss:
    
    namefile = f'../data/ngxs{ngxs}_nesf{n}_redshiftx.npz'
    stats = np.load(namefile)

    P0 = stats['arr_0']
    N_mean = stats['arr_1']
    xi_mean = stats['arr_2']
    rs = stats['arr_3']

    chi = -np.log(P0)/N_mean
    NE = N_mean*xi_mean

    plt.plot(NE,chi,ls='-',c=c)#,label=ngxs)

plt.xlim([1E-1,1E2])
plt.xscale('log')   
plt.legend(loc=3)
plt.show()

#%%
"""
Quiero hacer modificaciones al modelo teorico
binomial negativo y ver si se apega a la chi en z space
"""

plt.figure(figsize=(9,6))
plt.rcParams['font.size'] = 14

x = np.geomspace(1E-1,1E2,50)
n = 100000
c='k'
plt.plot(x,np.log(1+x*1)/(x*1),label='Negative Binomial',c=c)
a=.8
plt.plot(x,(1/((1-a)*(x/a)))*((1+x/a)**(1-a)-1),label='Generalized Hierarhichal',c=c,ls='--')

ngx=1000000
    
namefile = f'../data/ngxs{ngxs}_nesf{n}_redshiftx.npz'
stats = np.load(namefile)

P0 = stats['arr_0']
N_mean = stats['arr_1']
xi_mean = stats['arr_2']
rs = stats['arr_3']

chi = -np.log(P0)/N_mean
NE = N_mean*xi_mean

plt.plot(NE,chi,ls='-')#,label=ngxs)

plt.xlim([1E-1,1E2])
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
