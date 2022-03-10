from cProfile import label
from dataclasses import dataclass
import imp
from random import random
from traceback import print_tb

import matplotlib 
import matplotlib.pyplot as plt
import numpy as np
from torch import randn
'''
fig=plt.figure()    #empty figure without axes
b=np.matrix([[1,4],[7,8]])
b_asarray=np.asarray(b)
print(b)
plt.show()
'''

'''
np.random.seed(4654321)
data={
    'a':np.arange(50),
    'c':np.random.randint(0,50,50),
    'd':np.random.randn(50)
}
data['b']=data['a']+10*np.random.randn(50)
data['d']=np.abs(data['d'])*100
fig,axes=plt.subplots()
axes.scatter('a','b',c='c',s='d',data=data)
axes.set_xlabel('a')
axes.set_ylabel('b')
plt.show()
'''
'''
x=np.linspace(0,2,100)
fig,axes=plt.subplots()
axes.plot(x,x**2,label='quadratic')
axes.set_title("Simple Plot")
axes.legend()
plt.show()
'''
'''
x=np.linspace(0,2,100)
plt.figure(figsize=(5,2.7))
plt.plot(x,x**3,label='cubic')
plt.show()
'''
def my_plotter(axes,data1,data2,param_dict):
    out=axes.plot(data1,data2,**param_dict)
    return out
data1,data2,data3,data4=np.random.randn(4,100)
fig,(axes1,axes2)=plt.subplots(1,2,figsize=(5,2.7))
my_plotter(axes1,data1,data2,{'marker':'x'})
my_plotter(axes2,data3,data4,{'marker':'o'})
plt.show()
