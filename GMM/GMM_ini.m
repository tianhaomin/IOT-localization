function [ mu,m_sigma,mp ] = GMM_ini( data,n_center )

[m,n]=size(data);
[data_id,centers]=kmeans(data,n_center);
mu=centers;
mp=zeros(1,n_center);
m_sigma=zeros(n,n,n_center);

for i=1:n_center
    tem_id=(data_id==i);
    m_sigma(:,:,i)=sigma(data(tem_id,:));
    mp(i)=sum(tem_id)/m;
end

end

function sig=sigma(data)

[m,n]=size(data);
u=mean(data,1);
tem_data=data-repmat(u,m,1);

sig=zeros(n,n);
for k1=1:m
%     for k2=1:m 
    sig=sig+tem_data(k1,:)'*tem_data(k1,:);
%     end
end
sig=(sig+ 1E-5.*diag(ones(n,1)))/m;
end

