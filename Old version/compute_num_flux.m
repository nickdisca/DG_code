function [flux] = compute_num_flux(u,r,alpha,d1,d2,fact_bd,radius,points,eq_type)

uL=u;
uR=NaN*zeros(size(u));

for n=1:size(u,4)
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

if eq_type=="linear"
    fxL=uL; fxR=uR; fyL=uL; fyR=uR; 
elseif eq_type=="sphere"
    angle=pi/2;
    beta_x=2*pi*radius/(12*86400)*(cos(points(r+1+1:2*(r+1),:,:))*cos(angle)+sin(points((r+1)+1:2*(r+1),:,:)).*cos(points(1:r+1,:,:))*sin(angle));
    beta_y=-2*pi*radius/(12*86400)*sin(angle)*sin(points(1:r+1,:,:));
    fxL=beta_x.*uL; fxR=beta_x.*uR; fyL=beta_y.*uL; fyR=beta_y.*uR; 
elseif eq_type=="swe"
    
    g=0;
    
    fxL=NaN*zeros(size(u)); fyL=NaN*zeros(size(u));
    fxL(:,:,:,1)=uL(:,:,:,2); fyL(:,:,:,1)=uL(:,:,:,3);
    fxL(:,:,:,2)=uL(:,:,:,2).^2./uL(:,:,:,1)+g/2*uL(:,:,:,1).^2; fyL(:,:,:,2)=uL(:,:,:,2).*uL(:,:,:,3)./uL(:,:,:,1);
    fxL(:,:,:,3)=uL(:,:,:,2).*uL(:,:,:,3)./uL(:,:,:,1); fyL(:,:,:,3)=uL(:,:,:,3).^2./uL(:,:,:,1)+g/2*uL(:,:,:,1).^2;
    
    fxR=NaN*zeros(size(u)); fyR=NaN*zeros(size(u));
    fxR(:,:,:,1)=uR(:,:,:,2); fyR(:,:,:,1)=uR(:,:,:,3);
    fxR(:,:,:,2)=uR(:,:,:,2).^2./uR(:,:,:,1)+g/2*uR(:,:,:,1).^2; fyR(:,:,:,2)=uR(:,:,:,2).*uR(:,:,:,3)./uR(:,:,:,1);
    fxR(:,:,:,3)=uR(:,:,:,2).*uR(:,:,:,3)./uR(:,:,:,1); fyR(:,:,:,3)=uR(:,:,:,3).^2./uR(:,:,:,1)+g/2*uR(:,:,:,1).^2;

elseif eq_type=="swe_sphere"    

    fxL=NaN*zeros(size(u)); fyL=NaN*zeros(size(u)); g=0;
    fxL(:,:,:,1)=uL(:,:,:,2); fyL(:,:,:,1)=uL(:,:,:,3);
    fxL(:,:,:,2)=uL(:,:,:,2).^2./uL(:,:,:,1)+g/2*uL(:,:,:,1).^2; fyL(:,:,:,2)=uL(:,:,:,2).*uL(:,:,:,3)./uL(:,:,:,1);
    fxL(:,:,:,3)=uL(:,:,:,2).*uL(:,:,:,3)./uL(:,:,:,1); fyL(:,:,:,3)=uL(:,:,:,3).^2./uL(:,:,:,1)+g/2*uL(:,:,:,1).^2;
    
    fxR=NaN*zeros(size(u)); fyR=NaN*zeros(size(u));
    fxR(:,:,:,1)=uR(:,:,:,2); fyR(:,:,:,1)=uR(:,:,:,3);
    fxR(:,:,:,2)=uR(:,:,:,2).^2./uR(:,:,:,1)+g/2*uR(:,:,:,1).^2; fyR(:,:,:,2)=uR(:,:,:,2).*uR(:,:,:,3)./uR(:,:,:,1);
    fxR(:,:,:,3)=uR(:,:,:,2).*uR(:,:,:,3)./uR(:,:,:,1); fyR(:,:,:,3)=uR(:,:,:,3).^2./uR(:,:,:,1)+g/2*uR(:,:,:,1).^2;
    
else
    error('Undefinded equation type');
end

normal_x=zeros(size(u,1),size(u,2),4); normal_y=zeros(size(u,1),size(u,2),4);
normal_x(:,:,1)=0; normal_x(:,:,2)=1; normal_x(:,:,3)=0; normal_x(:,:,4)=-1;
normal_y(:,:,1)=-1; normal_y(:,:,2)=0; normal_y(:,:,3)=1; normal_y(:,:,4)=0;

flux=NaN*zeros(size(u));
for n=1:size(u,4)
flux(:,:,:,n)=1/2*((fxL(:,:,:,n)+fxR(:,:,:,n)).*normal_x+(fyL(:,:,:,n)+fyR(:,:,:,n)).*fact_bd.*normal_y)...
    -alpha/2*(uR(:,:,:,n)-uL(:,:,:,n)).*(normal_x.^2+fact_bd.*normal_y.^2);
end

%check flux at poles
%max(max(max(max(flux(:,d2:d2:d1*d2,3,:)))),max(max(max(flux(:,1:d2:(d1-1)*d2+1,1,:)))))

end