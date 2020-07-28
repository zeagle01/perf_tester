
import numpy as np
import matplotlib.pyplot as plt

file='build/result.txt'
with open(file,'r') as fin:
    words=fin.readline()
    words=words.split(' ')
    words=np.array(words)
    case_names=words[1:]



data=np.loadtxt(file)


fig,ax=plt.subplots(1,2)
for i,case_name in enumerate(case_names):
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set_xlabel('size(byte)')
    ax[0].set_ylabel('T(s)')

    data[:,i*2+1]/=1e6
    ax[0].plot(data[:,i*2+0],data[:,i*2+1],'-o',label=case_name)



    ax[1].set_xscale('log')
    ax[1].set_xlabel('size(byte)')
    ax[1].set_ylabel('bps(byte/s)')
    ax[1].plot(data[:,i*2+0],data[:,i*2+0]/data[:,i*2+1],'-o',label=case_name)


    ax[0].legend()
    ax[1].legend()


plt.show()
