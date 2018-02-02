
function [x1,x2,x3,value]=Powell

clc;clear all;clear
tic

%------------------------------------------预处理
% load d;load G_new;load x;
% x_origin=x;
% dd=d;
% clear d;
% clear x;
% global dd ;
% global G_new ;
% global x_origin ;




%-----------------------------------------------------------------------step1
N=1203;
e=10^-10;  %给定允许误差 
X0=0.005*rand(1,N);  %给定初始点
D=eye(N);   %矩阵D存放d1,d2,d3三个方向矢量
% D=[1 0 0;0 1 0;0 0 1]; 
while(1)
%-----------------------------------------------------------------------step2
    [X(1,:),fX(1)]=OneDimensionSearch(X0,D(1,:),N);
    for k=2:N               %从Xi出发，沿方向di进行搜索，得到最大值fXi,对应点为Xi
%         d(k) = D(k,:);
        [X(k,:),fX(k)]=OneDimensionSearch(X(k-1,:),D(k,:),N);
    end
    fX0 = Fx(1,X0,ones(1,N),N);  %计算fX0的值
    Diff(1)=fX(1)-fX0;
    for k=2:N
        Diff(k)=fX(k)-fX(k-1);
    end
    [maxDiff,m]=max(Diff);
 %-------------------------------------------------------------------step3
    d(N+1,:)=X(N,:)-X0;  %d4为这一轮产生的新方向向量
  %若距离(x[k,n]-x[k,0])<=e，则停止计算，得到结果XN
    temp1=X(N,:)-X0;
    Conditon1=sqrt(sum(temp1.*temp1));
    if Conditon1<=e
        break
    end
     %从点X0出发沿d(N+1)进行一轮搜索，得到最大值fX(N+1)，对应点X(N+1)及步长landa
    [X(N+1,:),fX(N+1),landa]=OneDimensionSearch(X0,d(N+1,:),N);
    X0=X(N+1,:);  %起始点X0换成X4
    %-------------------------------------------------------------------step4
    %若距离(x[k]-x[k-1])<=e，则停止计算，得点x[k]
    temp2=X(N+1,:)-X(N,:);
    Conditon2=sqrt(sum(temp2.*temp2));
    if Conditon2<=e
        X(N,:)=X(N+1,:);
        break;
    end
    temp3=sqrt((fX(N+1)-fX0)/(maxDiff+eps));
    %判断是否替换方向向量
    if(abs(landa)>temp3)
        %共轭性得到增强，替换选定的方向向量
        D(N+1,:)=d(N+1);
        for j=m:N
            D(j,:)=D(j+1,:);
        end
    end
    %否则，共轭性没有得到增强，不替换方向向量，重新进行搜索
end
for ii=1:N      %返回搜索结果
    x(ii)=X(N,ii);
end
value=fX(N)


load V
load R
L=401;
estR=x'*V';
figure
subplot(311);plot(estR(1:L));hold on;plot(R(1:L),'r');title('R_{p}');%axis([0 395 -0.02 0.02])
subplot(312);plot(estR(L+1:2*L));hold on;plot(R(L+1:2*L),'r');title('R_{s}');%axis([0 395 -0.02 0.02])
subplot(313);plot(estR(2*L+1:3*L));hold on;plot(R(2*L+1:3*L),'r');title('R_{d}');%axis([0 395 -0.01 0.01])
legend('est','ture')

toc






%一维搜索采用Brent
function[Y,fY,landa]=OneDimensionSearch(X,direction,N)
global dd ;
global G_new ;
global x_origin ;

a=-10;
b=10;
Epsilon=10^-10;
cgold=0.381966;
IterTimes=200;
if a>b
    temp=a;
    a=b;
    b=temp;
end
v=a+cgold*(b-a);
w=v;
x=v;
e=0.0;
fx=Fx(x,X,direction,N);
fv=fx;
fw=fx;
for iter=1:IterTimes
    xm=0.5*(a+b);
    if abs(x-xm)<=Epsilon*2-0.5*(b-a)
        break;
    end
    if abs(e)>Epsilon
        r=(x-w)*(fx-fv);
        q=(x-v)*(fx-fw);
        p=(x-v)*q-(x-w)*r;
        q=2*(q-r);
        if q>0
            p=-p;
        end
        q=abs(q);
        etemp=e;
        e=d;
        if not(abs(p)>=abs(0.5*q*etemp) || p<=q*(a-x) || p>=q*(b-x))
            d=p/q;
            u=x+d;
            if u-a<Epsilon*2 || b-u<Epsilon *2
                d=MySign(Epsilon,xm-x);
            end
        else
            if x>=xm
                e=a-x;
            else
                e=b-x;
            end
            d=cgold*e;
        end
    else
        if x>=xm
            e=a-x;
        else
            e=b-x;
        end
        d=cgold*e;
    end
    if abs(d)>=Epsilon
        u=x+d;
    else
        u=x+MySign(Epsilon,d);
    end
    fu=Fx(u,X,direction,N);
    if fu<=fx
        if u>=x
            a=x;
        else
            b=x;
        end
        v=w;
        fv=fw;
        w=x;
        fw=fx;
        x=u;
        fx=fu;
    else
        if u<x
            a=u;
        else
            b=u;
        end
        if fu<=fw ||w==x
            v=w;
            fv=fw;
            w=u;
            fw=fu; 
        else
            if fu<=fv ||v==x ||v==w
                v=u;
                fv=fu;
            end
        end
    end
end
landa=x;  %返回一维搜索结果
Y=X+x*direction;
fY=Fx(x,X,direction,N);



function [mySign]=MySign(a,b)
if b>0
    Result=abs(a);
else
    Result=-abs(a);
end
mySign=Result;




function [fx]=Fx(landa,X,direction,N)  %目标函数
global dd ;
global G_new ;
global x_origin ;

for i=1:N
    x(i)=X(i)+direction(i)*landa;
end
d_est=G_new*x';
Jd=0; 
for i=1:6015
    Jd=Jd+abs(dd(i)-d_est(i));
end
Jd=Jd/sum(abs(dd));

Jp=0;
for i=1:N
    Jp=Jp+abs(x(i)-x_origin(i));
end
fx=Jd+Jp;


% fx=(x(1)+10.123)^2+(x(2)-10.666)^2+(x(3)+1800.987)^4;
% x1=X(1)+direction(1)*landa;
% x2=X(2)+direction(2)*landa;
% x3=X(3)+direction(3)*landa;
% fx=(x1+10.123)^2+(x2-10.666)^2+(x3+1800.987)^4;







