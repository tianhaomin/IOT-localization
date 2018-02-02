function [fx]=Fx(landa,X,direction,N)  %Ä¿±êº¯Êý
global dd ;
global G_new ;
global x_origin ;
global W;

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
Jw=X*W*X';
fx=Jd+3*Jp+Jw;


% fx=(x(1)+10.123)^2+(x(2)-10.666)^2+(x(3)+1800.987)^4;
% x1=X(1)+direction(1)*landa;
% x2=X(2)+direction(2)*landa;
% x3=X(3)+direction(3)*landa;
% fx=(x1+10.123)^2+(x2-10.666)^2+(x3+1800.987)^4;