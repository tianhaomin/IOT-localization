function [mu,msigma,mp]=GMM(data,n_center,loglik_threshold)

[ mu,msigma,mp ] = GMM_ini( data,n_center );
disp('GMM_Ini Completed ! ');

Qt=E_step(data,mu,msigma,mp);
loglik_pre=loglike(data,mu,msigma,mp);
step=0;

while 1
   [mu,msigma,mp]=M_step(Qt,data);
   loglik_nxt=loglike(data,mu,msigma,mp);
  if abs((loglik_nxt/loglik_pre)-1) < loglik_threshold  
    break;  
  end
  
  if step>4
      break;
  end
  
  step=step+1;
  step
  loglik_pre=loglik_nxt;
  Qt=E_step(data,mu,msigma,mp);
  
  
end

end

function Qt=E_step(data,mu,m_sigma,mp)

n_model=length(mp);
m=size(data,1);
pxj(m,n_model)=0;

for j=1:n_model
   pxj(:,j)=GaussianPDF(data,mu(j,:),m_sigma(:,:,j));
end

px=pxj.*repmat(mp,m,1);
sp=sum(px,2);
Qt=px./repmat(sp,1,n_model);

end

function [lu,lsigma,lp]=M_step(Qt,data)

[m,n_model]=size(Qt);
n=size(data,2);

lu=zeros(n_model,n);
lsigma=zeros(n,n,n_model);
lp=zeros(1,n_model);

mul_data=zeros(n,n);

for j=1:n_model 
    lu(j,:)=sum(data.*repmat(Qt(:,j),1,n))/sum(Qt(:,j));   
    tem_data=data-repmat(lu(j,:),m,1); 
        for k=1:m
             mul_data=mul_data+tem_data(k,:)'*tem_data(k,:)*Qt(k,j); 
        end
    lsigma(:,:,j)=realmin+mul_data/sum(Qt(:,j));
    lp(j)=sum(Qt(:,j))/m;
end


end

function loglik=loglike(data,mu,msigma,mp)

n_center=size(mu,1);
pxj=zeros(size(data,1),n_center);
 for j=1:n_center  
    pxj(:,j) = GaussianPDF(data, mu(j,:), msigma(:,:,j));  
  end  
  F = pxj*mp';  
  F(F<realmin) = realmin;  
  loglik = log(sum(F)); 
  
end