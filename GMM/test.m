clear all;
clc;
data=rand(1000,128);
n_center=5;
thresh=0.0005;
[u,sigma,p]=GMM(data,n_center,thresh);

disp('Test Completed !');