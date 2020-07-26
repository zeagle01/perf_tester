
import numpy as np
import matplotlib.pyplot as plt

data=np.loadtxt('build/result.txt')


fig,ax=plt.subplots(1,2)
case_num=data.shape[1]//2
for i in range(case_num):
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')

    data[:,i*2+1]/=1e6
    ax[0].plot(data[:,i*2+0],data[:,i*2+1],'-o')

    ax[1].set_xscale('log')
    ax[1].plot(data[:,i*2+0],data[:,i*2+0]/data[:,i*2+1],'-o')

plt.show()
