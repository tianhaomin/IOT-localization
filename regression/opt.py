# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 13:11:00 2017

@author: Administrator
"""

import scipy.optimize as opt
import numpy as np

points=[]
def obj_func(p):
    x1,x2=p
    z=(1.74018092e+01)+(-2.41232312e-01)*x1+(-3.25819792e+00)*x2+(-1.67755970e-01)*x1**2+(9.46116968e-01)*x1*x2+(2.24755487e+00)*x2**2+(1.39952641e-03)*x1**3+(-3.32730684e-01)*x1**2*x2+(-4.67487295e-01)*x1*x2**2+(-7.29328135e-01)*x2**3+(7.80053382e-03)*x1**4+(4.66959057e-02)*x1**3*x2+(5.26926462e-02)*x1**2*x2**2+(4.61311364e-02)*x1*x2**3+(9.90805086e-02)*x2**4+(-6.23903410e-04)*x1**5+(-2.29681270e-03)*x1**4*x2+(-4.29366213e-04)*x1**3*x2**2+(-4.73212070e-03)*x1**2*x2**3+(-1.17877509e-04)*x1*x2**4+(-4.65569108e-03)*x2**5
    points.append((x1,x2,z))
    return z

#偏导数，有些优化方法用得到，有些用不到
def fprime(p):
    x1,x2=p
#    dx1=0.669684446*x1 - 0.00447399056*x2 - 3.10686268
#    dx2=-0.00447399056*x1 + 0.579336736*x2 - 3.2399659
    dx1=-0.00311951705*x1**4 - 0.0091872508*x1**3*x2 + 0.03120213528*x1**3 - 0.001288098639*x1**2*x2**2 + 0.1400877171*x1**2*x2 + 0.00419857923*x1**2 - 0.0094642414*x1*x2**3 + 0.1053852924*x1*x2**2 - 0.665461368*x1*x2 - 0.33551194*x1 - 0.000117877509*x2**4 + 0.0461311364*x2**3 - 0.467487295*x2**2 + 0.946116968*x2 - 0.241232312
    dx2= -0.0022968127*x1**4 - 0.000858732426*x1**3*x2 + 0.0466959057*x1**3 - 0.0141963621*x1**2*x2**2 + 0.1053852924*x1**2*x2 - 0.332730684*x1**2 - 0.000471510036*x1*x2**3 + 0.1383934092*x1*x2**2 - 0.93497459*x1*x2 + 0.946116968*x1 - 0.0232784554*x2**4 + 0.3963220344*x2**3 - 2.187984405*x2**2 + 4.49510974*x2 - 3.25819792
    return np.array([dx1,dx2])
def hess(p):
    x1,x2 = p
    
init_point=(1,1)

#这两种优化方法没用到偏导
result1=opt.fmin(obj_func,init_point)
result2=opt.fmin_powell(obj_func,init_point)

#用到偏导的：
result3=opt.fmin_cg(obj_func,init_point,fprime=fprime)
result4=opt.fmin_bfgs(obj_func,init_point,fprime=fprime)
result5=opt.fmin_tnc(obj_func,init_point,fprime=fprime)
result6=opt.fmin_l_bfgs_b(obj_func,init_point,fprime=fprime)

#其它
result7=opt.fmin_cobyla(obj_func,init_point,[])
##需要用到hessian矩阵的
print(result1,result2,result3,result4,result5,result6,result7)
import math
def er(a,b): 
    t = math.sqrt((a-4.7)**2+(b-5.8)**2)
    return t


e1=er(4.73496075 , 6.03049323)
e2=er(1.12812906e+64  , 1.50198046e+64)
e3=er(4.73495092 , 6.03044635)
e4=er(4.73495228 , 6.03044726)
e5=er(3.05957344e+23,   3.26857451e+25)
e6=er(1.89670372e+47,   4.98479440e+47)
e7=er(4.73490248 , 6.030491 )
e = [e1,e2,e3,e4,e5,e6,e7]
import seaborn
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
pm = plt.bar([1,2,3,4,5,6,7],e)
ax.set_xticks([1,2,3,4,5,6,7])
ax.set_xticklabels(['fmin','fmin_powell','fmin_cg','fmin_bfgs','fmin_tnc','fmin_l_bfgs_b','fmin_cobyla'])
ax.set_ylabel('ERRO(m)')
ax.set_xlabel('optmize')
#plt.ylim(0.17,0.173)
plt.show()