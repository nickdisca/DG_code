clear all; close all; clc;

a=0; b=2*pi; c=-pi/2; d=pi/2; %domain
d1=80; d2=80; %nb of elems in both directions
hx=(b-a)/d1; hy=(d-c)/d2;
r=1; %polynomial degree
dim=(r+1)^2;
eq_type="sphere"; %type of the equation

x_e=linspace(a,b,d1+1); y_e=linspace(c,d,d2+1);

unif=linspace(-1,1,r+1); unif=unif'; 
x_u=x_e(1:end-1)+(unif+1)/2*hx;
y_u=y_e(1:end-1)+(unif+1)/2*hy;
[unif2d,~]=tensor_product(unif,unif,NaN);
unif2d_phi=map2phi(unif2d,r,x_e,y_e,d1,d2,hx,hy);

[pts,wts]=gauss_legendre(r+1,-1,1);
%pts=flipud(pts); wts=flipud(wts); %order from -1 to 1
x=x_e(1:end-1)+(1+pts)/2*hx;
y=y_e(1:end-1)+(1+pts)/2*hy;

[pts2d,wts2d]=tensor_product(pts,pts,wts);

pts2d_phi=map2phi(pts2d,r,x_e,y_e,d1,d2,hx,hy);
pts2d_phi_bd=map2phi_bd(pts,r,x_e,y_e,d1,d2,hx,hy);

%vandermonde matrix to perform change of basis
V=NaN*zeros(dim,dim);
for i=1:dim
    for j=1:dim
        jj=floor((j-1)/(r+1))+1;
        jjj=mod(j-1,(r+1))+1;
        V(i,j)=legendreP(jj-1,unif2d(i,1))*legendreP(jjj-1,unif2d(i,2));
    end
end

%values and grads of basis functions in internal quadrature points
phi_val=NaN*zeros(dim,dim);
for i=1:dim
    for j=1:dim
        jj=floor((j-1)/(r+1))+1;
        jjj=mod(j-1,(r+1))+1;
        phi_val(i,j)=legendreP(jj-1,pts2d(i,1))*legendreP(jjj-1,pts2d(i,2));
    end
end
phi_grad=NaN*zeros(dim,dim,2);
for i=1:dim
    for j=1:dim
        jj=floor((j-1)/(r+1))+1;
        jjj=mod(j-1,(r+1))+1;
        phi_grad(i,j,1)=1/sqrt((2*(jj-1)+1)/2)*GradJacobiP(pts2d(i,1),0,0,jj-1)*legendreP(jjj-1,pts2d(i,2));
        phi_grad(i,j,2)=1/sqrt((2*(jjj-1)+1)/2)*GradJacobiP(pts2d(i,2),0,0,jjj-1)*legendreP(jj-1,pts2d(i,1));
    end
end
%values of basis functions in boundary quadrature points
phi_val_bd=NaN*zeros(r+1,dim,4);
for i=1:r+1
    for j=1:dim
        jj=floor((j-1)/(r+1))+1;
        jjj=mod(j-1,(r+1))+1;
        phi_val_bd(i,j,2)=legendreP(jj-1,1)*legendreP(jjj-1,pts(i)); %right
        phi_val_bd(i,j,4)=legendreP(jj-1,-1)*legendreP(jjj-1,pts(i)); %left
        phi_val_bd(i,j,1)=legendreP(jj-1,pts(i))*legendreP(jjj-1,-1); %bottom
        phi_val_bd(i,j,3)=legendreP(jj-1,pts(i))*legendreP(jjj-1,1); %top
    end
end

%compute factor (cosine for sphere, 1 for cartesian)
[fact_int,fact_bd,complem_fact,radius]=compute_factor(eq_type,r,d1,d2,pts2d_phi(dim+1:2*dim,:),pts2d_phi_bd((r+1)+1:2*(r+1),:,:));

%initial condition (cartesian linear advection)
if eq_type=="linear"
    u0=sin(2*pi*unif2d_phi(1:dim,:)).*sin(2*pi*unif2d_phi(dim+1:2*dim,:));
    %u0=unif2d_phi(1:dim,:);
    %u0=5*ones(size(unif2d_phi(1:dim,:)));
end

%initial condition (spherical linear advection)
if eq_type=="sphere"
    th_c=0; lam_c=3/2*pi; h0=1000; 
    rr=radius*acos(sin(th_c)*sin(unif2d_phi(dim+1:2*dim,:))+...
        cos(th_c)*cos(unif2d_phi(dim+1:2*dim,:)).*cos(unif2d_phi(1:dim,:)-lam_c)); 
    u0=h0/2*(1+cos(pi*rr/radius)).*(rr<radius/3);
    %u0=ones(size(rr));
end

%initial condition (cartesian swe)
if eq_type=="swe"
    h0=1000; h1=5; L=1e7; sigma=L/20;
    u0(:,:,1)=h0+h1*exp(-((unif2d_phi(1:dim,:)-L/2).^2+(unif2d_phi(dim+1:2*dim,:)-L/2).^2)/(2*sigma^2));
    u0(:,:,2)=zeros(size(u0(:,:,1)));
    u0(:,:,3)=zeros(size(u0(:,:,1)));
end

%initial condition (spherical swe)
if eq_type=="swe_sphere"
    h0=2.94e4/9.80616; g=9.80616; Omega=7.292e-5; uu0=2*pi*radius/(12*86400); angle=0;
    u0(:,:,1)=h0-1/g*(radius*Omega*uu0+uu0^2/2)*(sin(unif2d_phi(dim+1:2*dim,:))*cos(angle)-cos(unif2d_phi(dim+1:2*dim,:)).*cos(unif2d_phi(1:dim,:))*sin(angle)).^2;
    u0(:,:,2)=u0(:,:,1).*uu0.*(cos(unif2d_phi(dim+1:2*dim,:))*cos(angle)+sin(unif2d_phi(dim+1:2*dim,:)).*cos(unif2d_phi(1:dim,:))*sin(angle));
    u0(:,:,3)=-u0(:,:,1).*uu0.*sin(angle).*sin(unif2d_phi(1:dim,:));
    
    %u0(:,:,1)=h0*ones(dim,d1*d2);
    %u0(:,:,2)=zeros(size(u0(:,:,1)));
    %u0(:,:,3)=zeros(size(u0(:,:,1)));
    
    %u0(:,:,1)=h0*unif2d_phi(dim+1:2*dim,:);
    %u0(:,:,2)=zeros(size(u0(:,:,1)));
    %u0(:,:,3)=zeros(size(u0(:,:,1)));
end

%visualize solution at initial time
figure; plot_solution(u0(:,:,1),x_u,y_u,r,d1,d2);
%figure; plot_solution_fine(u0(:,:,1),V,x_e,y_e,2*r,d1,d2);

%convert nodal to modal
u=u0; 
for i=1:size(u,3), u(:,:,i)=V\u(:,:,i); end

%mass matrix
mass=compute_mass(phi_val,r,wts2d,d1,d2,hx,hy,fact_int);

%temporal loop
t=0;
dt=12; %time step
T=12*86400; %final time
RK=4; %order of the RK scheme
N_it=round(T/dt);
if eq_type=="linear", velocity=1; Courant=dt*velocity/min(hx,hy); end
if eq_type=="sphere", velocity=2*pi*radius/(12*86400); Courant=dt*velocity/(radius*min(hx,hy)); end
if eq_type=="swe", velocity=sqrt(9.80616*h0); Courant=dt*velocity/min(hx,hy); end
if eq_type=="swe_sphere", velocity=sqrt(sqrt(2.94e4)+2*pi*radius/(12*86400)); Courant=dt*velocity/(radius*min(hx,hy)); end
fprintf('Time integration: order %d, T=%f, dt=%f, N_iter=%d, Courant=%f\n',RK,T,dt,N_it,Courant);

for i=1:N_it

    if RK==1
        u=u+dt*compute_rhs(u,r,mass,phi_val,phi_grad,phi_val_bd,hx,hy,wts,wts2d,d1,d2,fact_int,fact_bd,complem_fact,radius,pts2d_phi,pts2d_phi_bd,eq_type);
    end

    if RK==2
        k1=compute_rhs(u,r,mass,phi_val,phi_grad,phi_val_bd,hx,hy,wts,wts2d,d1,d2,fact_int,fact_bd,complem_fact,radius,pts2d_phi,pts2d_phi_bd,eq_type);
        k2=compute_rhs(u+dt*k1,r,mass,phi_val,phi_grad,phi_val_bd,hx,hy,wts,wts2d,d1,d2,fact_int,fact_bd,complem_fact,radius,pts2d_phi,pts2d_phi_bd,eq_type);
        u=u+dt*1/2*k1+dt*1/2*k2;   
    end
    
    if RK==3
        k1=compute_rhs(u,r,mass,phi_val,phi_grad,phi_val_bd,hx,hy,wts,wts2d,d1,d2,fact_int,fact_bd,complem_fact,radius,pts2d_phi,pts2d_phi_bd,eq_type);
        k2=compute_rhs(u+dt*k1,r,mass,phi_val,phi_grad,phi_val_bd,hx,hy,wts,wts2d,d1,d2,fact_int,fact_bd,complem_fact,radius,pts2d_phi,pts2d_phi_bd,eq_type);
        k3=compute_rhs(u+dt*(1/4*k1+1/4*k2),r,mass,phi_val,phi_grad,phi_val_bd,hx,hy,wts,wts2d,d1,d2,fact_int,fact_bd,complem_fact,radius,pts2d_phi,pts2d_phi_bd,eq_type);
        u=u+dt*1/6*k1+dt*1/6*k2+dt*2/3*k3;   
    end
    
    if RK==4
        k1=compute_rhs(u,r,mass,phi_val,phi_grad,phi_val_bd,hx,hy,wts,wts2d,d1,d2,fact_int,fact_bd,complem_fact,radius,pts2d_phi,pts2d_phi_bd,eq_type);
        k2=compute_rhs(u+dt*k1/2,r,mass,phi_val,phi_grad,phi_val_bd,hx,hy,wts,wts2d,d1,d2,fact_int,fact_bd,complem_fact,radius,pts2d_phi,pts2d_phi_bd,eq_type);
        k3=compute_rhs(u+dt*(1/2*k2),r,mass,phi_val,phi_grad,phi_val_bd,hx,hy,wts,wts2d,d1,d2,fact_int,fact_bd,complem_fact,radius,pts2d_phi,pts2d_phi_bd,eq_type);
        k4=compute_rhs(u+dt*(1*k3),r,mass,phi_val,phi_grad,phi_val_bd,hx,hy,wts,wts2d,d1,d2,fact_int,fact_bd,complem_fact,radius,pts2d_phi,pts2d_phi_bd,eq_type);
        u=u+dt*1/6*k1+dt*1/3*k2+dt*1/3*k3+dt*1/6*k4;   
    end
    
    if RK>=5
        error('RK scheme not implemented');
    end
    
    if(mod(i,100)==0), pause(0.5); plot_solution(V*u(:,:,1),x_u,y_u,r,d1,d2); fprintf('Iteration %d/%d\n',i,N_it); end
    
    if t+dt>T, dt=T-t; end; t=t+dt;
    if max(abs(u(:)))>=1e8, fprintf('Iteration %d!!!\n',i); error('Solution is diverging'); end
    
end

%for i=1:size(u,3), figure; plot_solution(V*u(:,:,i),x_u,y_u,r,d1,d2); end
for i=1:size(u,3), figure; uu=V*u(:,:,i); plot_solution_fine(uu,unif2d_phi,r,d1,d2); end

errL2=0; normL2_sol=0;
for i=1:d1*d2, errL2=errL2+(u(:,i,1)-V\u0(:,i,1))'*mass(:,:,i)*(u(:,i,1)-V\u0(:,i,1)); end
for i=1:d1*d2, normL2_sol=normL2_sol+(V\u0(:,i,1))'*mass(:,:,i)*(V\u0(:,i,1)); end
fprintf('L2 error is %f, and after normalization is %f\n',sqrt(errL2),sqrt(errL2/normL2_sol));


fprintf('End of program\n');