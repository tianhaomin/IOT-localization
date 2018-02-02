# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 14:31:20 2017

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

#position = simulation(100,2.4,3.5,1)
#x1,x2,z,x = decome(position)
def draw_surface(x1,x2,z):
    fig = plt.figure()  
    ax = fig.add_subplot(111, projection='3d') 
    ax.scatter(x1,x2,z)
    ax.plot_trisurf(x1,x2,z)
#    ax.set_xlabel("x")
#    ax.set_xlabel("y")
#    ax.set_xlabel("Pj-P1")
    plt.show()

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
            ff=np.array([x_r[i],y_r[i]]).reshape(1,2)
            if reg.predict(poly.fit_transform(ff))< temp:
                temp = reg.predict(poly.fit_transform(ff))     
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

def MLP_regressor(x,z,layers,batchsize=3,max_iter=5):
    reg = mlpr(hidden_layer_sizes=layers,solver = 'adam',batch_size =batchsize,learning_rate='adaptive',max_iter=max_iter)
    reg.fit(x,z)
    return reg

def svressor(x,z,kernel='rbf'):
    clf = svm.SVR(kernel='rbf')
    clf.fit(x,z) 
    return clf
    
def KNN_Ressor(x,z,k):
    neigh = KNeighborsRegressor(n_neighbors=k)
    neigh.fit(x,z)
    return neigh

def RN_Ressor(x,z,r=1):
    neigh = RadiusNeighborsRegressor(radius=r)
    neigh.fit(x,z)
    return neigh

def DescitionTree_Ressor(x,z):
    clf = tree.DecisionTreeRegressor()
    clf = clf.fit(x, z)
    return clf

def SGDRessor(x,z):
    clf = linear_model.SGDRegressor()
    clf.fit(x, z)
    return clf

def comparation(x,z,c1,num_search):
    r1=linear_regressor(x,z)
    r2=MLP_regressor(x,z,(5,5))
    r3=svressor(x,z)
    r4=KNN_Ressor(x,z,3)
    r5=RN_Ressor(x,z)
    r6=DescitionTree_Ressor(x,z)
    r7=SGDRessor(x,z)
    r = [r2,r3,r4,r5,r6,r7]
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
    dis = []
    for i in range(6):
        q,w,e = search(x_l,x_h,y_l,y_h,r[i],num_search,ispoly=False)
        d = math.sqrt((w-c1[0])**2 + (e-c1[1])**2)
        dis.append(d)
    q,w,e = search(x_l,x_h,y_l,y_h,r1,num_search,ispoly=True)
    d = math.sqrt((w-c1[0])**2 + (e-c1[1])**2)
    dis.append(d)
    print(dis)
    fig, ax = plt.subplots()
    pm = plt.bar([1,2,3,4,5,6,7],dis)
    ax.set_xticks([1,2,3,4,5,6,7])
    ax.set_xticklabels(['NN','SVR','KNN','R_NN','DTR','SGDR','LR(3)'])
    ax.set_ylabel('ERRO(m)')
    ax.set_xlabel('regression')
    plt.show()


def order_compare(center):
    c1 = center
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
    error = []
    for i in range(20):
        reg = linear_regressor(x,z,k=i)
        q_p,w_p,e_p = search(x_l,x_h,y_l,y_h,reg,100,ispoly=True,k=i)
        d = math.sqrt((w_p-c1[0])**2 + (e_p-c1[1])**2)
        error.append(d)
    return error


if __name__ == "__main__":
###b不同的回归方法的对比
    position = simulation (150,4.7,5.8)
    x1,x2,z,x = decome(position)
    comparation(x,z,[4.7,5.8],100)
#########多项式拟合次数的优化
    c1 = [4.7,5.8]
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
    error = []
    for i in range(20):
        reg = linear_regressor(x,z,k=i)
        q_p,w_p,e_p = search(x_l,x_h,y_l,y_h,reg,100,i,ispoly=True)
        d = math.sqrt((w_p-c1[0])**2 + (e_p-c1[1])**2)
        error.append(d)
    x_ = np.arange(1,21,1)
    plt.plot(x_,np.array(error))
    plt.xlabel('order of poly')
    plt.ylabel('Error(m)')
    plt.show()       
#    error_r = []
#    error_p = []
#    for i in range(5):
#        x1 = random.uniform(0,10)
#        y1 = random.uniform(0,10)
#        center = [x1,y1]
#        error = order_compare(center)
#        error_r.append(error)
#    for i in range(20):
#        error_p.append((error_r[0][i]+error_r[1][i]+error_r[2][i]+error_r[3][i]+error_r[4][i])/5.0)
#    x_ = np.arange(1,21,1)
#    plt.scatter(x_,np.array(error_p))
#    plt.xlabel('order of poly')
#    plt.ylabel('error(m)')
        
###绘制仿真曲面
    draw_surface(x1,x2,z)
###nodes密度的对比
    nodes = [4,10,20,50,100,150,200]
    c1 = [4.23,5.45]
    error = []
    for i in nodes:
        position = simulation (i,4.23,5.45)
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
###搜索次数对定位精度的影响
    ser_num = [10,30,50,100,200,500,1000]
    c1 = [4.23,5.45]
    position = simulation (50,4.23,5.45)
    x1,x2,z,x = decome(position)
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
    error = []
    reg = linear_regressor(x,z,k=5)
    for i in ser_num:
        q_p,w_p,e_p = search(x_l,x_h,y_l,y_h,reg,i,ispoly=True,k=5)
        d = math.sqrt((w_p-c1[0])**2 + (e_p-c1[1])**2)
        error.append(d)
    plt.plot(ser_num,np.array(error))
    plt.xlabel('num of serch')
    plt.ylabel('Error(m)')
    plt.show()       
###nodes数目和多项式次数的结合判断
    nodes = [10,20,50,100,200]
    error_z = []
    for node in nodes:
        error_t  =[]
        position = simulation (node,4.7,5.8)
        x1,x2,z,x = decome(position)
        c1 = [4.7,5.8]
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
        for i in range(3,7):
            reg = linear_regressor(x,z,k=i)
            q_p,w_p,e_p = search(x_l,x_h,y_l,y_h,reg=reg,ispoly=True,num_search=100,k=i)
            d = math.sqrt((w_p-c1[0])**2 + (e_p-c1[1])**2)
            error_t.append(d)
        error_z.append(error_t)
        data = error_z
        dim = len(data[0])
        w = 0.75
        dimw = w / dim
        x = np.arange(len(data))
        fig, ax = plt.subplots()
        #x = np.array(nodes)
        for i in range(len(data[0])):
            y = [d[i] for d in data]
            b = plt.bar(x + i * dimw, y, dimw, bottom=0.001,label='order=%d'%(i+3))
        plt.xticks(x + dimw / 2, map(str, x))
        plt.xlabel('nodes')
        plt.ylabel('Error(m)')
        ax.set_xticklabels(['10', '20', '50','100','200'])
        plt.legend()
        plt.show()
####绘制拟合曲面
    position = simulation (150,0.5,0.7)
    x1,x2,z,x = decome(position)
    draw_surface(x1,x2,z)
    reg = linear_regressor(x,z,k=3)
    par = reg.coef_
    x = np.array(x1)
    y = np.array(x2)
    z = (par[0])+(par[1])*x+(par[2])*y+(par[3])*x**2+(par[4])*x*y+(par[5])*y**2+(par[6])*x**3+(par[7])*x**2*y+(par[8])*x*y**2+(par[9])*y**3
    fig = plt.figure()  
    ax = fig.add_subplot(111, projection='3d') 
#    ax.scatter(x,y,z)
    ax.plot_trisurf(list(x),list(y),list(z))
#    ax.set_xlabel("x")
#    ax.set_xlabel("y")
#    ax.set_xlabel("Pj-P1")
    plt.show()
##干扰源位置不同进行比对
#    nodes = [[0.5,0.5],[0.8,0.8],[1.0,1.0],[1.2,1.2],[1.7,1.7],[1.8,1.8]]
#    nodes=[[9.5,0.5],[9.3,0.7],[9,1.0],[8.9,1.2],[8.7,1.7],[8.8,1.8]]
#    nodes =  [[0.5,4.3],[0.8,4.2],[1.0,4.6],[1.2,4.1],[1.7,4.7],[1.8,4.5]]
    nodes = [[4.5,4.3],[4.8,4.2],[4.0,4.6],[4.2,4.1],[4.7,4.7],[4.5,4.5]]
    error = []
    for i in range(len(nodes)):
        c1=nodes[i]
        position = simulation (50,c1[0],c1[1])
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
    fig, ax = plt.subplots()
    bar_width = 0.35
    index = np.arange(6)
    ax.set_xticks(index + bar_width / 2)
    ax.bar(index,error)
    plt.xlabel('position of nodes')
    plt.ylabel('Error(m)')
    ax.set_xticklabels(('A1', 'B1', 'C1', 'D1', 'E1','F1','G1','H1','I1'))
    plt.show()
    







