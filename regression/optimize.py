# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 21:44:09 2017

@author: Administrator
"""
import numpy as np
from scipy import optimize
import math
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sympy import *
from scipy.optimize import basinhopping
def simulation (nodes_num,c_x,c_y,threshold=1,pj=10,k=0,n=2):
    c1 = [c_x,c_y]
    position = []
    dim = int(math.sqrt(nodes_num))
    delta = 10/dim
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

def linear_regressor(x,z,k):
    poly = PolynomialFeatures(degree=k)
    x_new = poly.fit_transform(x)
    reg = LinearRegression(fit_intercept=False)
    reg.fit(x_new,z)
    return reg

x = Symbol('x')
y = Symbol('y')
#####优化算法
def target_function(x,y):
    return (2.20116058e+01)-(3.10686268*x)-(3.23996590*y)+(3.34842223e-01*x**2)-(4.47399056e-03*x*y)+(2.89668368e-01*y**2)
def dif(fun):
    x = Symbol('x')
    y = Symbol('y')
    return diff(fun,x),diff(fun,y)
#    
class TargetFunction(object):
    def __init__(self):
        self.f_points = []
        self.fprime_points = []
        self.fhess_points = []
    def f(self,p):
        x,y = p.tolist()
        z = target_function(x,y)
        self.f_points.append((x,y))
        return z
    def fprime(self,p):##求导
        x,y = p.tolist()
        self.fprime_points.append((x,y))
        dx,dy = dif(self.f(p))
        return np.array([dx,dy])
    def fhess(self,p):##求hessian矩阵
        x,y = p.tolist()
        self.fhess_points.append((x,y))
        dx,dy = dif(self.f(p))
        dxdx = dif(dx,x)
        dxdy = dif(dx,y)
        dydy = dif(dy,y)
        return np.array([[dxdx,dxdy],[dxdy,dydy]])

def fmin_demo(method):
    target = TargetFunction()
    init_point=(1,1)
    res = optimize.minimize(target.f,init_point,method=method,
                            jac=target.fprime,hess=target.fhess)
    return res,[np.array(points) for points in (target.f_points,target.fprime_points,target.fhess_points)]

mothods = ("Nelder-Mead","Powell","CG","BFGS","Newton-CG","L-BFGS-B")
for method in mothods:
    res,(f_points,fprime_points,fhess_points) = fmin_demo(method)
    print("{:12s}: min={:12g},f count={:3d},fprime count={:3d},"\
          "fhess count={:3d}".format(method,float(res["fun"]),len(f_points),len(fprime_points),len(fhess_points)))


position = simulation (150,4.7,5.8)
x1,x2,z,x = decome(position)
reg = linear_regressor(x,z,k=5)
para = reg.coef_
poly = PolynomialFeatures(degree=5)
poly.fit_transform(x)
print(poly.get_feature_names("xy"))




    