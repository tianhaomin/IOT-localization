# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 19:43:06 2017

@author: Administrator
"""
##线性回归
import numpy as np
import math
import heapq
import random
k=5
c1 = [2.7,8.1]
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


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly = PolynomialFeatures(degree=k) #1\x1\x2\x1**2\x1*x2\x2**2
x_new = poly.fit_transform(x)
reg = LinearRegression(fit_intercept=False)
reg.fit(x_new,z)
parameters = reg.coef_
reg.predict(poly.fit_transform(c1))

result = heapq.nlargest(4, z)
index = []
for i in range(len(z)):
    if z[i] in result:
        index.append(i)
result_point = [x[i] for i in index]
###开始蒙特卡洛搜索
def search(x_l,x_h,y_l,y_h,num):
    poly = PolynomialFeatures(degree=k)
    x_r = np.arange(x_l,x_h,(x_h-x_l)/float(num))
    y_r = np.arange(y_l,y_h,(y_h-y_l)/float(num))
    x_r = list(x_r)
    y_r = list(y_r)
    temp = 0
    x_temp = x_l
    y_temp = y_l 
    for i in range(num):
        print('ff')
        if reg.predict(poly.fit_transform([x_r[i],y_r[i]])) > temp:
            temp = reg.predict(poly.fit_transform([x_r[i],y_r[i]]))     
            x_temp = x_r[i]
            y_temp = y_r[i]
        else:
            print(i)
            pass
    return temp,x_temp,y_temp

q,w,e = search(2,3,4,6,1000)

d2 = math.sqrt((w-c1[0])**2 + (e-c1[1])**2)

reg.coef_










