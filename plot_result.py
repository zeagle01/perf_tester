
import numpy as np
import matplotlib.pyplot as plt

file='build/result.txt'
with open(file,'r') as fin:
    words=fin.readline()
    words=words.split(' ')
    words=np.array(words)
    case_names=words[1:]



data=np.loadtxt(file)


record_field_num=3
duration_col=2

fig,ax=plt.subplots(1,2)
for i,case_name in enumerate(case_names):
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set_xlabel('problem size (byte)')
    ax[0].set_ylabel('time consumed (s)')

    data[:,i*record_field_num+duration_col]=(data[:,i*record_field_num+duration_col]+0.1)/1e6
    ax[0].plot(data[:,i*record_field_num+0],data[:,i*record_field_num+duration_col],'-o',label=case_name)



    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_xlabel('problem size (byte)')
    ax[1].set_ylabel('operation count per second ($s^{-1}$)')
    ax[1].plot(data[:,i*record_field_num+0],data[:,i*record_field_num+1]/data[:,i*record_field_num+duration_col],'-o',label=case_name)


    ax[0].legend()
    ax[1].legend()


plt.show()
