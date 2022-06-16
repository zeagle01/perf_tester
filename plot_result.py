
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
problem_size_in_byte=0
operation_size_in_byte=1
duration=2

fig,ax=plt.subplots(2,2)
ax=[ax[0,0],ax[0,1],ax[1,0],ax[1,1]]


label_x= [
    '',
    '',
    'problem size (byte)',
    'problem size (byte)',
]

label= [
    'time consumed (s)',
    'throughput (Gfloat/s)',
    'operation per float',
    'G operations per second',
]

for i,case_name in enumerate(case_names):

    problem_size=data[:,i*record_field_num+problem_size_in_byte]/4
    duration_in_second=(data[:,i*record_field_num+duration]+0.1)/1e6
    operation_size=data[:,i*record_field_num+operation_size_in_byte]/4

    ax[0].plot(problem_size,duration_in_second,'-o',label=case_name)

    ax[1].plot(problem_size,problem_size/duration_in_second/1e9,'-o',label=case_name)

    ax[2].plot(problem_size,operation_size/problem_size,'-o',label=case_name)

    ax[3].plot(problem_size,operation_size/duration_in_second/1e9,'-o',label=case_name)




for i,a in enumerate(ax):
    a.set_xscale('log')
    a.set_yscale('log')
    a.set_xlabel(label_x[i])
    a.set_ylabel(label[i])
    a.grid()
    a.legend()


plt.show()
