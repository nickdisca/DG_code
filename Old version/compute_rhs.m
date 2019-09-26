function [rhsu] = compute_rhs(u,r,mass,phi_val,phi_grad,phi_val_bd,hx,hy,wts,wts2d,d1,d2,fact_int,fact_bd,complem_fact,radius,pts2d_phi,pts2d_phi_bd,eq_type)

rhsu=zeros(size(u));

determ=hx*hy/4;
bd_det(1)=hx/2; bd_det(2)=hy/2; bd_det(3)=hx/2; bd_det(4)=hy/2;

u_qp=zeros(size(u));
for i=1:size(u,3), u_qp(:,:,i)=phi_val*u(:,:,i); end
dim=(r+1)^2;

%internal integrals
if eq_type=="linear"
    flux_fun_x=u_qp; flux_fun_y=u_qp;
elseif eq_type=="sphere"
    angle=pi/2; 
    beta_x=2*pi*radius/(12*86400)*(cos(pts2d_phi(dim+1:2*dim,:))*cos(angle)+sin(pts2d_phi(dim+1:2*dim,:)).*cos(pts2d_phi(1:dim,:))*sin(angle));
    beta_y=-2*pi*radius/(12*86400)*sin(angle)*sin(pts2d_phi(1:dim,:));
    flux_fun_x=beta_x.*u_qp; flux_fun_y=beta_y.*u_qp;
elseif eq_type=="swe"
    g=0;
    flux_fun_x(:,:,1)=u_qp(:,:,2); 
    flux_fun_y(:,:,1)=u_qp(:,:,3);
    flux_fun_x(:,:,2)=u_qp(:,:,2).^2./u_qp(:,:,1)+g/2*u_qp(:,:,1).^2; 
    flux_fun_y(:,:,2)=u_qp(:,:,2).*u_qp(:,:,3)./u_qp(:,:,1);
    flux_fun_x(:,:,3)=u_qp(:,:,2).*u_qp(:,:,3)./u_qp(:,:,1); 
    flux_fun_y(:,:,3)=u_qp(:,:,3).^2./u_qp(:,:,1)+g/2*u_qp(:,:,1).^2;
elseif eq_type=="swe_sphere"
    g=0;
    flux_fun_x(:,:,1)=u_qp(:,:,2); 
    flux_fun_y(:,:,1)=u_qp(:,:,3);
    flux_fun_x(:,:,2)=u_qp(:,:,2).^2./u_qp(:,:,1)+g/2*u_qp(:,:,1).^2; 
    flux_fun_y(:,:,2)=u_qp(:,:,2).*u_qp(:,:,3)./u_qp(:,:,1);
    flux_fun_x(:,:,3)=u_qp(:,:,2).*u_qp(:,:,3)./u_qp(:,:,1); 
    flux_fun_y(:,:,3)=u_qp(:,:,3).^2./u_qp(:,:,1)+g/2*u_qp(:,:,1).^2;
else
    error('Undefinded equation type');
end

for i=1:size(u,3)
    rhsu(:,:,i)=phi_grad(:,:,1)'*(flux_fun_x(:,:,i).*wts2d)*(2/hx)*determ +...
        phi_grad(:,:,2)'*(fact_int.*flux_fun_y(:,:,i).*wts2d)*(2/hy)*determ;
end

%fluxes
u_qp_bd=zeros(length(wts),size(u,2),4,size(u,3));
for n=1:size(u,3)
    for i=1:4, u_qp_bd(:,:,i,n)=phi_val_bd(:,:,i)*u(:,:,n); end
end

if eq_type=="linear"
    alpha=1; 
elseif eq_type=="sphere"
    alpha=2*pi*radius/(12*86400);
elseif eq_type=="swe"
    alpha=sqrt(9.80616*1000);
elseif eq_type=="swe_sphere"
    alpha=sqrt(2.94e4)+2*pi*radius/(12*86400);
else
    error('Undefinded equation type');
end

num_flux=compute_num_flux(u_qp_bd,r,alpha,d1,d2,fact_bd,radius,pts2d_phi_bd,eq_type);

%boundary integrals
bd_term=zeros(size(u));
for n=1:size(u,3)
    for i=1:4
        bd_term(:,:,n)=bd_term(:,:,n)+phi_val_bd(:,:,i)'*(num_flux(:,:,i,n).*wts)*bd_det(i);
    end
    rhsu(:,:,n)=rhsu(:,:,n)-bd_term(:,:,n);
end

%add nonconservative term at rhs of SWE -g*h*grad(eta)
if eq_type=="swe_sphere" || eq_type=="swe"
    
    g=9.80616;
    grad_eta_x=(2/hx)*phi_grad(:,:,1)*u(:,:,1); %zero topography
    grad_eta_y=(2/hy)*phi_grad(:,:,2)*u(:,:,1); %zero topography
    rhsu(:,:,2)=rhsu(:,:,2)-g*phi_val(:,:)'*(u_qp(:,:,1).*grad_eta_x.*wts2d)*determ;
    rhsu(:,:,3)=rhsu(:,:,3)-g*phi_val(:,:)'*(fact_int.*u_qp(:,:,1).*grad_eta_y.*wts2d)*determ;
    
    num_flux_nc=compute_num_flux_nc(u_qp_bd,r,alpha,d1,d2,fact_bd,radius,pts2d_phi_bd,eq_type);
    bd_term=zeros(size(u));
    for n=2:size(u,3)
        for i=1:4
            bd_term(:,:,n)=bd_term(:,:,n)+g*phi_val_bd(:,:,i)'*(num_flux_nc(:,:,i,n).*wts)*bd_det(i);
        end
        rhsu(:,:,n)=rhsu(:,:,n)-bd_term(:,:,n);
    end
    
end

%add corrective internal term for the divergence, only in spherical coordinates for systems
if eq_type=="swe_sphere"
    rhsu(:,:,2)=rhsu(:,:,2)+phi_val(:,:)'*(complem_fact.*flux_fun_y(:,:,2).*wts2d)*determ;
    rhsu(:,:,3)=rhsu(:,:,3)-phi_val(:,:)'*(complem_fact.*flux_fun_x(:,:,2).*wts2d)*determ;
end

%add coriolis term
if eq_type=="swe"
	coriolis=0;
    rhsu(:,:,2)=rhsu(:,:,2)+radius*phi_val(:,:)'*(fact_int.*coriolis.*u_qp(:,:,3).*wts2d)*determ;
	rhsu(:,:,3)=rhsu(:,:,3)-radius*phi_val(:,:)'*(fact_int.*coriolis.*u_qp(:,:,2).*wts2d)*determ;
end
if eq_type=="swe_sphere"
    angle=0;
	coriolis=2*7.292e-5*(sin(pts2d_phi(dim+1:2*dim,:))*cos(angle)-cos(pts2d_phi(dim+1:2*dim,:)).*cos(pts2d_phi(1:dim,:))*sin(angle));
    %coriolis=0;
    rhsu(:,:,2)=rhsu(:,:,2)+radius*phi_val(:,:)'*(fact_int.*coriolis.*u_qp(:,:,3).*wts2d)*determ;
	rhsu(:,:,3)=rhsu(:,:,3)-radius*phi_val(:,:)'*(fact_int.*coriolis.*u_qp(:,:,2).*wts2d)*determ;
end

for n=1:size(u,3)
    for i=1:d1*d2, rhsu(:,i,n)=1/radius*(mass(:,:,i)\rhsu(:,i,n)); end
end

end