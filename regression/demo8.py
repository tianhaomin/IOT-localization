# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 09:05:33 2017

@author: Administrator
"""
import matplotlib.pyplot as plt
import seaborn
d = [d1,d2,d3,d4,d5,d6,d7]
fig, ax = plt.subplots()
pm = plt.bar([1,2,3,4,5,6,7],d)
ax.set_xticks([1,2,3,4,5,6,7])
ax.set_xticklabels(['NN','LR(5)','SVR','KNN','R_NN','DTR','SGDR'])
ax.set_ylabel('ERRO(m)')
ax.set_xlabel('regression')
plt.show()

##对于SVR采用不同的核函数

import matplotlib.pyplot as plt
import seaborn
d = [0.065,1.12,1.11,1.12]
fig, ax = plt.subplots()
pm = plt.bar([1,2,3,4],d)
ax.set_xticks([1,2,3,4])
ax.set_xticklabels(['rbf','linear','poly','sigmoid'])
ax.set_ylabel('ERRO(m)')
ax.set_xlabel('kernel')
plt.show()







