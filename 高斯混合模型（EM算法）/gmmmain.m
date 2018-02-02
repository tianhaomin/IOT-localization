clc;clear all;close all
%产生2个二维正态数据
MU1    = [1 2 3];
SIGMA1 = [1 0 0.5;0 1 0.4;0.5 0.4 0.5];
MU2    = [-4 2 7];
SIGMA2 = [1 0 0.5;0 1 0.4;0.5 0.4 0.5];
X      = [mvnrnd(MU1, SIGMA1, 1000);mvnrnd(MU2, SIGMA2, 1000)];
%GMMs学习
[Data,Mu0,w0,Sigma0] = gmmsinit();  
[w, Mu, Sigma] = gmmsEM(Data, w0, Mu0, Sigma0) 

