function [flux] = compute_num_flux_nc(u,r,alpha,d1,d2,fact_bd,radius,points,eq_type)

uL=u;
uR=NaN*zeros(size(u));

for n=1:1 %no need to repeat for all components
    ids=d2+1:(d1-1)*d2; 
    uR(:,ids,2,n)=u(:,ids+d2,4,n); uR(:,ids,4,n)=u(:,ids-d2,2,n); %flux lr, internal
    uR(:,(d1-1)*d2+1:d1*d2,4,n)=u(:,ids(end-d2+1:end),2,n); %flux l, right bd
    uR(:,1:d2,2,n)=u(:,d2+1:2*d2,4,n); %flux r, left bd
    uR(:,(d1-1)*d2+1:d1*d2,2,n)=u(:,1:d2,4,n); uR(:,1:d2,4,n)=u(:,(d1-1)*d2+1:d1*d2,2,n); %periodic
    
    ids=1:d1*d2; ids(~mod(ids,d2))=[]; ids(~mod(ids-1,d2))=[];
    uR(:,ids,3,n)=u(:,ids+1,1,n); uR(:,ids,1,n)=u(:,ids-1,3,n); %flux tb, internal
    uR(:,1:d2:(d1-1)*d2+1,3,n)=u(:,2:d2:(d1-1)*d2+2,1,n); %flux t, bottom bd
    uR(:,d2:d2:d1*d2,1,n)=u(:,d2-1:d2:d1*d2-1,3,n); %flux b, top bd
    uR(:,d2:d2:d1*d2,3,n)=u(:,1:d2:(d1-1)*d2+1,1,n); uR(:,1:d2:(d1-1)*d2+1,1,n)=u(:,d2:d2:d1*d2,3,n); %periodic
end

if eq_type=='swe_sphere' || eq_type=='swe'  
    
    normal_x=zeros(size(u,1),size(u,2),4); normal_y=zeros(size(u,1),size(u,2),4);
    normal_x(:,:,1)=0; normal_x(:,:,2)=1; normal_x(:,:,3)=0; normal_x(:,:,4)=-1;
    normal_y(:,:,1)=-1; normal_y(:,:,2)=0; normal_y(:,:,3)=1; normal_y(:,:,4)=0;
    
    flux=NaN*zeros(size(u));
    %(eta^*-eta)*h^star*n
    flux(:,:,:,2)=((uR(:,:,:,1)+uL(:,:,:,1))/2-uL(:,:,:,1)).*(uR(:,:,:,1)+uL(:,:,:,1))/2.*normal_x;
    flux(:,:,:,3)=((uR(:,:,:,1)+uL(:,:,:,1))/2-uL(:,:,:,1)).*(uR(:,:,:,1)+uL(:,:,:,1))/2.*normal_y.*fact_bd;
    
else
    error('Trying to compute NC term for a non SWE equation');
end


end