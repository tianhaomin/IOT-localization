    function sig=sigma(data)%计算初始化的方差  
      
      
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