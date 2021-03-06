function [x,val,k]=bfgs(fun,gfun,x0)
%功能：用BFGS算法求解无约束问题：min f(x)
% 输入：x0是初始点，fun,gfun分别是目标函数及其梯度；
%varargin是输入可变参数变量，简单调用bfgs时可以忽略它，
% 但是其他程序循环调用时将会发挥重要作用
%输出：x,val分别是近似最优点和最优值，k是迭代次数。

maxk=20;       %给出最大迭代次数
rho=0.55; sigma=0.4; epsilon=1e-5;      %给出一些常数参数及精度误差
k=0; n=length(x0);
Bk=eye(n);      %Bk=feval('Hesse',x0);

while(k<maxk)
    gk=feval(gfun,x0);      %计算精度
    if(norm(gk)<epsilon)
        break;
    end                         %检验终止准则
    dk=-Bk\gk;      %解方程组，计算搜索方向
    m=0;mk=0;
    while(m<20)     %用Armijo搜索求步长
        newf=feval(fun,x0+rho^m*dk)
        mk=m;
        break;
    end
    m=m+1;
end

%BFGS矫正
x=x0+rho^mk*dk;
sk=x-x0;  yk=feval(gfun,x)-gk;
if(yk'*sk>0)
    Bk=Bk-(Bk*sk*sk'*Bk)/(sk'*Bk*sk)+(yk*yk')/(yk'*sk);
end
k=k+1;      x0=x;
val=feval(fun,x0);
end

