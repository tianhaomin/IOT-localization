# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 21:10:41 2017

@author: Administrator
"""
import numpy as np
import math
import heapq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
import seaborn
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor as mlpr
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn import tree
from sklearn import linear_model
import random
def simulation (nodes_num,c_x,c_y,threshold=1,pj=10,k=0,n=2):
    c1 = [c_x,c_y]
    position = []
    dim = int(math.sqrt(nodes_num))
    delta = 18.3/dim
    for i in range(dim):
        for j in range(dim):
            position.append([i*delta,j*delta])
    for point in position:
        if math.sqrt((point[0]-c1[0])**2+(point[1] - c1[1])**2) < threshold:
            position.remove(point)
        else:
            pass
    for point in position:
        distant = math.sqrt((point[0]-c1[0])**2+(point[1] - c1[1])**2)
        point.append(distant)
    for point in position:
        point[2] = 10*n*math.log10(point[2])
    
    return position
#position = simulation(100,2.4,3.5,1)
def decome(list1):
    x1 = []
    x2 = []
    z = []
    for i in list1:
        x1.append(i[0])
        x2.append(i[1])
        z.append(i[2])
    x = [[x1[i],x2[i]] for i in range(len(x1))] 
    x = np.array(x)
    return x1,x2,z,x

def search(x_l,x_h,y_l,y_h,reg,num_search,k,ispoly=True):
    x_r = []
    y_r = []
    if ispoly == True:
        poly = PolynomialFeatures(degree=k)
        if x_l == x_h and y_l != y_h:
            x_r = np.ones(num_search)*x_h
            y_r = np.arange(y_l,y_h,(y_h-y_l)/float(num_search))
        if y_l == y_h and x_l != x_h:
            y_r = np.ones(num_search)*y_h
            x_r = np.arange(x_l,x_h,(x_h-x_l)/float(num_search))
        if x_l != x_h and y_l != y_h:
            x_r = np.arange(x_l,x_h,(x_h-x_l)/float(num_search))
            y_r = np.arange(y_l,y_h,(y_h-y_l)/float(num_search))
        if x_l == x_h and y_l == y_h:
            x_r = np.ones(num_search)*x_h
            y_r = np.ones(num_search)*y_h
        x_r = list(x_r)
        y_r = list(y_r)
        temp = 100
        x_temp = x_l
        y_temp = y_l 
        for i in range(num_search):
            #print('ff')
            ff=np.array([[x_r[i]],[y_r[i]]])
            if reg.predict(poly.fit_transform(ff))< temp:
                temp = reg.predict(poly.fit_transform([x_r[i],y_r[i]]))     
                x_temp = x_r[i]
                y_temp = y_r[i]
            else:
                #print(i)
                pass
        return temp,x_temp,y_temp
    else:
        if x_l == x_h and y_l != y_h:
            x_r = np.ones(num_search)*x_h
            y_r = np.arange(y_l,y_h,(y_h-y_l)/float(num_search))
        if y_l == y_h and x_l != x_h:
            y_r = np.ones(num_search)*y_h
            x_r = np.arange(x_l,x_h,(x_h-x_l)/float(num_search))
        if x_l != x_h and y_l != y_h:
            x_r = np.arange(x_l,x_h,(x_h-x_l)/float(num_search))
            y_r = np.arange(y_l,y_h,(y_h-y_l)/float(num_search))
        if x_l == x_h and y_l == y_h:
            x_r = np.ones(num_search)*x_h
            y_r = np.ones(num_search)*y_h
        x_r = list(x_r)
        y_r = list(y_r)
        temp = 100000
        x_temp = x_l
        y_temp = y_l 
        for i in range(num_search):
            #print('ff')
            if reg.predict([x_r[i],y_r[i]]) < temp:
                temp = reg.predict([x_r[i],y_r[i]])
                x_temp = x_r[i]
                y_temp = y_r[i]
            else:
                #print(i)
                pass
        return temp,x_temp,y_temp

def linear_regressor(x,z,k):
    poly = PolynomialFeatures(degree=k)
    x_new = poly.fit_transform(x)
    reg = LinearRegression(fit_intercept=False)
    reg.fit(x_new,z)
    return reg


position = simulation (150,4.7,5.8)
x1,x2,z,x = decome(position)

nodes = [4,10,20,50,100,150,200]
c1 = [4.7,5.8]
error = []
position = simulation (100,4.23,5.45)
x1,x2,z,x = decome(position)
reg = linear_regressor(x,z,k=5)
result = heapq.nsmallest(4, z)
index = []
for i in range(len(z)):
    if z[i] in result:
        index.append(i)
result_point = [x[i] for i in index]
x_list = []
y_list = []
for i in result_point:
    x_list.append(i[0])
    y_list.append(i[1])
x_l = min(x_list)
x_h = max(x_list)
y_l = min(y_list)
y_h = max(y_list)
q_p,w_p,e_p = search(x_l,x_h,y_l,y_h,reg,100,5,ispoly=True)
d = math.sqrt((w_p-c1[0])**2 + (e_p-c1[1])**2)
error.append(d)
plt.plot(nodes,error)
plt.xlabel('num of nodes')
plt.ylabel('Error(m)')
plt.show()


 x_r = []
    y_r = []

        poly = PolynomialFeatures(degree=5)
        if x_l == x_h and y_l != y_h:
            x_r = np.ones(100)*x_h
            y_r = np.arange(y_l,y_h,(y_h-y_l)/float(100))
        if y_l == y_h and x_l != x_h:
            y_r = np.ones(100)*y_h
            x_r = np.arange(x_l,x_h,(x_h-x_l)/float(100))
        if x_l != x_h and y_l != y_h:
            x_r = np.arange(x_l,x_h,(x_h-x_l)/float(100))
            y_r = np.arange(y_l,y_h,(y_h-y_l)/float(100))
        if x_l == x_h and y_l == y_h:
            x_r = np.ones(100)*x_h
            y_r = np.ones(100)*y_h
        x_r = list(x_r)
        y_r = list(y_r)
        temp = 100
        x_temp = x_l
        y_temp = y_l 
        for i in range(num_search):
            #print('ff')
            ff=np.array([x_r[0],y_r[0]]).reshape(1,2)
            if reg.predict(poly.fit_transform(ff))< temp:
                temp = reg.predict(poly.fit_transform(ff)    ) 
                x_temp = x_r[i]
                y_temp = y_r[i]



plt.plot([4,10,20,50,100,150,200],[6.028,1.9,2.26,0.99,0.34,0.41,0.46],label="Fitting five times")
plt.plot([4,10,20,50,100,150,200],[31,25,17.5,10,3.5,2,1],label='CL')
plt.plot([4,10,20,50,100,150,200],[18,16,14,8,3.2,1.6,0.76],label="GCL")
plt.xlabel('num of nodes')
plt.ylabel("Error(m)")
plt.legend()
plt.show()



