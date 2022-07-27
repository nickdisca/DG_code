function [flux] = compute_numerical_flux(u,d1,d2,fact_bd,eq_type,radius,qp_x,qp_y)

%uL is the solution in current element
uL=u;
%uR is the solution in the neighboring element
uR=nan(size(u));

for n=1:size(u,4)
    %left and right
    %'internal' indexes, i.e. exclude first and last column
    ids=d2+1:(d1-1)*d2; 
    %retrieve function from neighboring element
    uR(:,ids,2,n)=u(:,ids+d2,4,n); 
    uR(:,ids,4,n)=u(:,ids-d2,2,n); 
    %left face (index 4) on the right physical boundary (last column)
    uR(:,(d1-1)*d2+1:d1*d2,4,n)=u(:,ids(end-d2+1:end),2,n); 
    %right face (index 2) on the left physical boundary (first column)
    uR(:,1:d2,2,n)=u(:,d2+1:2*d2,4,n); 
    %apply periodic BC
    uR(:,(d1-1)*d2+1:d1*d2,2,n)=u(:,1:d2,4,n); 
    uR(:,1:d2,4,n)=u(:,(d1-1)*d2+1:d1*d2,2,n); 
    
    %top and bottom
    %'internal' indexes, i.e. exclude first and last row
    ids=1:d1*d2; ids(~mod(ids,d2))=[]; ids(~mod(ids-1,d2))=[];
    %retrieve function from neighboring element    
    uR(:,ids,3,n)=u(:,ids+1,1,n); 
    uR(:,ids,1,n)=u(:,ids-1,3,n); 
    %top face (index 3) on the bottom physical boundary (first row)
    uR(:,1:d2:(d1-1)*d2+1,3,n)=u(:,2:d2:(d1-1)*d2+2,1,n); 
    %bottom face (index 1) on the top physical boundary (last row)
    uR(:,d2:d2:d1*d2,1,n)=u(:,d2-1:d2:d1*d2-1,3,n); 
    %apply periodic BC
    uR(:,d2:d2:d1*d2,3,n)=u(:,1:d2:(d1-1)*d2+1,1,n); 
    uR(:,1:d2:(d1-1)*d2+1,1,n)=u(:,d2:d2:d1*d2,3,n);
end

[fxL, fyL] = flux_function(uL,eq_type,radius,qp_x,qp_y, fact_bd);
[fxR, fyR] = flux_function(uR,eq_type,radius,qp_x,qp_y, fact_bd);

%compute normal vectors for all faces: 1=bottom, 2=right, 3=top, 4=left
normal_x=zeros(size(u,1),size(u,2),4); 
normal_y=zeros(size(u,1),size(u,2),4);
normal_x(:,:,1)=0; normal_x(:,:,2)=1; normal_x(:,:,3)=0; normal_x(:,:,4)=-1;
normal_y(:,:,1)=-1; normal_y(:,:,2)=0; normal_y(:,:,3)=1; normal_y(:,:,4)=0;

%compute maximum wave speed for all faces
alpha = max( get_maximum_eig(uL,eq_type,radius,qp_x,qp_y), get_maximum_eig(uR,eq_type,radius,qp_x,qp_y));
% global alpha;
%compute Lax-Friedrichs flux (multiplied with normal vector)
flux=nan(size(u));
for n=1:size(u,4)
    flux(:,:,:,n)=1/2*((fxL(:,:,:,n)+fxR(:,:,:,n)).*normal_x+(fyL(:,:,:,n)+fyR(:,:,:,n)).*fact_bd.*normal_y)...
        -alpha/2.*(uR(:,:,:,n)-uL(:,:,:,n)).*(normal_x.^2+fact_bd.*normal_y.^2);
end

end