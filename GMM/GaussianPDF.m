function gp=GaussianPDF(data,u,sigma)

[m,n]=size(data);

pre_item=1/sqrt(((2*pi)^n)*abs(det(sigma)+realmin));
nxt_item(1:m)=0;
tem_data=data-repmat(u,m,1);
for i=1:m
   tem_data_t=tem_data(i,:)';
   nxt_item(i)=exp(-0.5*(tem_data(i,:)*(inv(sigma))*tem_data_t)); 
end

gp=pre_item*nxt_item;

end