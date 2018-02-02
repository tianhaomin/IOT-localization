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

