# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 21:50:47 2017

@author: Administrator
"""
#用神经网络进行拟合数据点进行预测
#单一干扰源存在仿真
from sklearn import mixture 
import numpy as np
import math
import heapq
import random
c1 = [2.5,5]
position = []
for i in range(10):
    for j in range(10):
        position.append([i,j])
for point in position:
    if math.sqrt((point[0]-c1[0])**2+(point[1] - c1[1])**2) < 1:
        position.remove(point)
    else:
        pass
for point in position:
    distant = math.sqrt((point[0]-c1[0])**2+(point[1] - c1[1])**2)
    point.append(distant)
#dis = []
#for point in position:
#    dis.append(point[2])
#max_dis = max(dis)
# 生成丢包率
for point in position:
    point[2] = math.exp(-point[2])

#new = []
#for i in range(len(position)):
#    new.append(position[i][2])

x1 = []
x2 = []
z = []
for i in position:
    x1.append(i[0])
    x2.append(i[1])
    z.append(i[2])
x = [[x1[i],x2[i]] for i in range(len(x1))] 
x = np.array(x)
###线性回归模型y = w1*x1+w2*x2 ridge 回归肯定是不对的
#x = [[x1[i],x2[i]] for i in range(99)] 
#from sklearn.linear_model import Ridge
#from sklearn.preprocessing import PolynomialFeatures
#from sklearn.linear_model import Perceptron
#clf = Ridge(alpha=1.0)
#clf.fit(x,z) 
#clf.coef_
### 多项式 拟合
#from sklearn.linear_model import Perceptron
#from sklearn.preprocessing import PolynomialFeatures
#x = [[x1[i],x2[i]] for i in range(99)]
#x = np.array(x)
#poly = PolynomialFeatures(degree=2)
#x_t = poly.fit_transform(x)
#clf = Perceptron(fit_intercept=False,shuffle=False).fit(x_t, z)
from sklearn.neural_network import MLPRegressor as mlpr
reg = mlpr(hidden_layer_sizes=(5,5),solver = 'adam',batch_size =3,learning_rate='adaptive',max_iter=5)
reg.fit(x,z)


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
import seaborn
fig = plt.figure()  
ax = fig.add_subplot(111, projection='3d') 
ax.scatter(x1,x2,z)
ax.plot_trisurf(x1,x2,z)
plt.show()

#reg.get_params(deep=True)
#parameter = reg.coefs_
#p = parameter[0].dot(parameter[1])
#p = p.dot(parameter[2])
#np.array([5,5]).dot(p)
#inv = np.linalg.pinv(p)

result = heapq.nlargest(4, z)
index = []
for i in range(len(z)):
    if result[0] == z[i]:
        index.append(i)
result_point = [x[i] for i in index]
###开始蒙特卡洛搜索
def search(x_l,x_h,y_l,y_h,num):
    x_r = np.arange(x_l,x_h,(x_h-x_l)/float(num))
    y_r = np.arange(y_l,y_h,(y_h-y_l)/float(num))
    x_r = list(x_r)
    y_r = list(y_r)
    temp = 0
    x_temp = x_l
    y_temp = y_l 
    for i in range(num):
        print('ff')
        if reg.predict([x_r[i],y_r[i]]) > temp:
            temp = reg.predict([x_r[i],y_r[i]])
            x_temp = x_r[i]
            y_temp = y_r[i]
        else:
            print(i)
            pass
    return temp,x_temp,y_temp

q,w,e = search(2,3,5,6.1,1000)

d1 = math.sqrt((w-c1[0])**2 + (e-c1[1])**2)



