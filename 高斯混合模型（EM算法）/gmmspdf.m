function prob = gmmspdf(X,Mu,Sigma)
[N,d] = size(X);
X = X - repmat(Mu',N,1);
prob = sum((X*inv(Sigma)).*X, 2);
prob = exp(-0.5*prob) / sqrt((2*pi)^d * (abs(det(Sigma))+realmin));
end