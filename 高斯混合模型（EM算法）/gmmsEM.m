function [w, Mu, Sigma] = gmmsEM(Data, w0, Mu0, Sigma0)
%EM迭代停止条件
loglik_threshold = 1e-10;
%初始化参数
[N, dim] = size(Data);
M = size(Mu0,2);
loglik_old = -realmax;
Step = 0;
Mu = Mu0;
Sigma = Sigma0;
w = w0;
Epsilon = 0.0001;
while (Step < 1000)
    Step = Step+1;
%E-步骤
for i=1:M
    Pxi(:,i) = gmmspdf(Data, Mu(:,i), Sigma(:,:,i));         
end
Pix_tmp = repmat(w,[N 1]).*Pxi;
Pix = Pix_tmp ./ (repmat(sum(Pix_tmp,2),[1 M])+realmin);
Beta = sum(Pix);
%M-步骤
for i=1:M
    w(i) = Beta(i) / N;
    Mu(:,i) = Data'*Pix(:,i) / Beta(i);
    Data_tmp1 = Data' - repmat(Mu(:,i),1,N);
    Sigma(:,:,i) = (repmat(Pix(:,i)',dim, 1) .* Data_tmp1*Data_tmp1') / Beta(i);
    Sigma(:,:,i) = Sigma(:,:,i) + 1E-5.*diag(ones(dim,1));
end
%终止条件
v = [sum(abs(Mu - Mu0)), abs(w - w0)];
s = abs(Sigma-Sigma0);
v2 = 0;
for i=1:M
    v2 = v2 + det(s(:,:,i));
end
if ((sum(v) + v2) < Epsilon)
    break;
end
Mu0 = Mu;
Sigma0 = Sigma;
w0 = w;
end
end