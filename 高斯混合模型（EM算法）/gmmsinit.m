%初始化参数  
function [Data,Mu0,w0,Sigma0] = gmmsinit()
MU1    = [1 2 3];
SIGMA1 = [1 0 0.5;0 1 0.4;0.5 0.4 0.5];
MU2    = [-1 -1 4];
SIGMA2 = [1 0 0.5;0 1 0.4;0.5 0.4 0.5];
X      = [mvnrnd(MU1, SIGMA1, 1000);mvnrnd(MU2, SIGMA2, 1000)];
Data = X;
Mu0 = [MU1' MU2'];
Sigma0(:,:,1) = SIGMA1;
Sigma0(:,:,2) = SIGMA2;
w0 = zeros(1,3); 
[N, dim] = size(Data);
K = size(Mu0,1);
distmat = repmat(sum(Data.*Data, 2), 1, K) + ... 
            repmat(sum(Mu0.*Mu0, 2)', N, 1) - ...
            2*Data*Mu0;
        [~, labels] = min(distmat, [], 2);
for k=1:K
    Xk = Data(labels == k, : );% Xk是所有被归到K类的X向量构成的矩阵
    w0(k) = size(Xk, 1)/N;
end