clear all; colors=get(gca,'ColorOrder'); close all; clc;
set(0,'defaultAxesFontSize',20); set(0,'defaultLineLineWidth',2);

%definition of the domain [a,b]x[c,d]
a=0; b=1; c=0; d=1;

%number of elements in x and y direction
d1=20; d2=20; 

%length of the 1D intervals
hx=(b-a)/d1; hy=(d-c)/d2;

%polynomial degree of DG
r=2; 

%cardinality
dim=(r+1)^2;

%equation type
eq_type="linear";

%type of quadrature rule (Gauss-Legendre or Gauss-Legendre-Lobatto)
quad_type="leg";

%number of quadrature points in one dimension
n_qp_1D=(r+1);

%number of quadrature points
n_qp=n_qp_1D^2;

%time interval, initial and final time
t=0;
T=1*(b-a);
%T=36000;
%T=12*86400;

%order of the RK scheme (1,2,3,4)
RK=3; 

%time step
dt=1/r^2*min(hx,hy)*0.1; 
%dt=100;
%dt=50;
%dt=0.01;

%beginning and end of the intervals
x_e=linspace(a,b,d1+1); 
y_e=linspace(c,d,d2+1);

%create uniformly spaced points in reference and physical domain
unif=linspace(-1,1,r+1)';
[unif2d,~]=tensor_product(unif,unif,nan(size(unif)));
unif2d_phi=map2phi(unif2d,r,x_e,y_e,d1,d2,hx,hy);

%create quadrature points and weights in reference and physical domain,
%both internally and on the boundary of the elements
if quad_type=="leg"
    [pts,wts]=gauss_legendre(n_qp_1D,-1,1); pts=flipud(pts); wts=flipud(wts);
elseif quad_type=="lob"
    [pts,wts]=gauss_legendre_lobatto(n_qp_1D-1); 
end
[pts2d,wts2d]=tensor_product(pts,pts,wts);
pts2d_phi=map2phi(pts2d,n_qp_1D-1,x_e,y_e,d1,d2,hx,hy);
pts2d_phi_bd=map2phi_bd(pts,n_qp_1D-1,x_e,y_e,d1,d2,hx,hy);

%Vandermonde matrix, needed to switch from modal to nodal: u_nod=V*u_mod,
%i.e. V(i,j)=Phi_j(x_i) for i,j=1:dim. The x_i are the uniform points
V=nan(dim,dim);
for i=1:dim
    for j=1:dim
        j_x=floor((j-1)/(r+1))+1;
        j_y=mod(j-1,(r+1))+1;
        V(i,j)=JacobiP(unif2d(i,1),0,0,j_x-1)*JacobiP(unif2d(i,2),0,0,j_y-1);
    end
end

%values and grads of basis functions in internal quadrature points, i.e.
%phi_val(i,j)=Phi_j(x_i) for i=1:dim_qp,j=1:dim. The x_i are the quadrature points,
%the gradient has two components due to the x and y derivatives
phi_val=nan(n_qp,dim);
for i=1:n_qp
    for j=1:dim
        j_x=floor((j-1)/(r+1))+1;
        j_y=mod(j-1,(r+1))+1;
        phi_val(i,j)=JacobiP(pts2d(i,1),0,0,j_x-1)*JacobiP(pts2d(i,2),0,0,j_y-1);
    end
end
phi_grad=nan(n_qp,dim,2);
for i=1:n_qp
    for j=1:dim
        j_x=floor((j-1)/(r+1))+1;
        j_y=mod(j-1,(r+1))+1;
        phi_grad(i,j,1)=GradJacobiP(pts2d(i,1),0,0,j_x-1)*JacobiP(pts2d(i,2),0,0,j_y-1);
        phi_grad(i,j,2)=GradJacobiP(pts2d(i,2),0,0,j_y-1)*JacobiP(pts2d(i,1),0,0,j_x-1);
    end
end

%values of basis functions in boundary quadrature points, repeating for
%each face
%dimensions: (num_quad_pts_per_face)x(cardinality)x(num_faces)
phi_val_bd=nan(n_qp_1D,dim,4);
for i=1:n_qp_1D
    for j=1:dim
        j_x=floor((j-1)/(r+1))+1;
        j_y=mod(j-1,(r+1))+1;
        %right
        phi_val_bd(i,j,2)=JacobiP(1,0,0,j_x-1)*JacobiP(pts(i),0,0,j_y-1);
        %left
        phi_val_bd(i,j,4)=JacobiP(-1,0,0,j_x-1)*JacobiP(pts(i),0,0,j_y-1);
        %bottom
        phi_val_bd(i,j,1)=JacobiP(pts(i),0,0,j_x-1)*JacobiP(-1,0,0,j_y-1); 
        %top
        phi_val_bd(i,j,3)=JacobiP(pts(i),0,0,j_x-1)*JacobiP(1,0,0,j_y-1);
    end
end

%compute factor (cosine for sphere, 1 for cartesian)
[fact_int,fact_bd,complem_fact,radius]=compute_factor(eq_type,pts2d_phi(n_qp+1:2*n_qp,:),pts2d_phi_bd(n_qp_1D+1:2*n_qp_1D,:,:));

%initial condition (cartesian linear advection) - specified as function
if eq_type=="linear"
    %u0_fun=@(x,y) sin(2*pi*x).*sin(2*pi*y);
    h0=0; h1=1; R=(b-a)/2/5; u0_fun=@(x,y) h0+h1/2*(1+cos(pi*sqrt((x-(a+b)/2).^2+(y-(c+d)/2).^2)/R)).*(sqrt((x-(a+b)/2).^2+(y-(c+d)/2).^2)<R);
    %u0_fun=@(x,y) 5*ones(size(x));
    
    %set initial condition in the uniformly spaced quadrature points
    u0=u0_fun(unif2d_phi(1:dim,:),unif2d_phi(dim+1:2*dim,:));

%initial condition (cartesian swe) - specified as function
elseif eq_type=="swe"
    h0=1000; h1=5; L=1e7; sigma=L/20;
    h0_fun=@(x,y) h0+h1*exp(-((x-L/2).^2+(y-L/2).^2)/(2*sigma^2));
    v0x_fun=@(x,y) zeros(size(x));
    v0y_fun=@(x,y) zeros(size(x));
    
    %set initial condition in the uniformly spaced quadrature points
    u0(:,:,1)=h0_fun(unif2d_phi(1:dim,:),unif2d_phi(dim+1:2*dim,:));
    u0(:,:,2)=h0_fun(unif2d_phi(1:dim,:),unif2d_phi(dim+1:2*dim,:)).*v0x_fun(unif2d_phi(1:dim,:),unif2d_phi(dim+1:2*dim,:));
    u0(:,:,3)=h0_fun(unif2d_phi(1:dim,:),unif2d_phi(dim+1:2*dim,:)).*v0y_fun(unif2d_phi(1:dim,:),unif2d_phi(dim+1:2*dim,:));
    
    %set coriolis function
    coriolis_fun=@(x,y) 1e-4*ones(size(x));
    
%initial condition (spherical linear advection) - specified as function
elseif eq_type=="adv_sphere"
    th_c=pi/2; lam_c=3/2*pi; h0=1000; 
    rr=@(lam,th) radius*acos(sin(th_c)*sin(th)+cos(th_c)*cos(th).*cos(lam-lam_c)); 
    u0_fun=@(lam,th) h0/2*(1+cos(pi*rr(lam,th)/radius)).*(rr(lam,th)<radius/3);
    
    %set initial condition in the uniformly spaced quadrature points
    u0=u0_fun(unif2d_phi(1:dim,:),unif2d_phi(dim+1:2*dim,:));  
    
%initial condition (spherical swe) - specified as function
elseif eq_type=="swe_sphere"
    g=9.80616; h0=2.94e4/g; Omega=7.292e-5; uu0=2*pi*radius/(12*86400); angle=0;
    h0_fun=@(lam,th) h0-1/g*(radius*Omega*uu0+uu0^2/2)*(sin(th)*cos(angle)-cos(lam).*cos(th)*sin(angle)).^2;
    v0x_fun=@(lam,th) uu0.*(cos(th)*cos(angle)+sin(th).*cos(lam)*sin(angle));
    v0y_fun=@(lam,th) -uu0.*sin(angle).*sin(lam);
    
    %set initial condition in the uniformly spaced quadrature points
    u0(:,:,1)=h0_fun(unif2d_phi(1:dim,:),unif2d_phi(dim+1:2*dim,:));
    u0(:,:,2)=h0_fun(unif2d_phi(1:dim,:),unif2d_phi(dim+1:2*dim,:)).*v0x_fun(unif2d_phi(1:dim,:),unif2d_phi(dim+1:2*dim,:));
    u0(:,:,3)=h0_fun(unif2d_phi(1:dim,:),unif2d_phi(dim+1:2*dim,:)).*v0y_fun(unif2d_phi(1:dim,:),unif2d_phi(dim+1:2*dim,:));
    
    %set coriolis function
    coriolis_fun=@(lam,th) 2*Omega*(sin(th)*cos(angle)-cos(th).*cos(lam)*sin(angle));
    
end

%if coriolis function is not defined (possibly because not needed), set it
%to zero
if ~exist('coriolis_fun','var')
    coriolis_fun=@(x,y) zeros(size(x));
end


%visualize solution at initial time - only first component
x_u=x_e(1:end-1)+(unif+1)/2*hx;
y_u=y_e(1:end-1)+(unif+1)/2*hy;
figure; plot_solution(u0(:,:,1),x_u(:),y_u(:),r,d1,d2,"contour");


%convert nodal to modal: the vector u will contain the modal coefficient
u=u0; 
for i=1:size(u,3)
    u(:,:,i)=V\u(:,:,i); 
end

%compute mass matrix
mass=compute_mass(phi_val,wts2d,d1,d2,hx,hy,fact_int);

%temporal loop parameters
Courant=dt/min(hx,hy);
N_it=ceil(T/dt);

%print information
fprintf('Space discretization: order %d, elements=%d*%d, domain=[%f,%f]x[%f,%f]\n',r,d1,d2,a,b,c,d);
fprintf('Time integration: order %d, T=%f, dt=%f, N_iter=%d\n',RK,T,dt,N_it);

%start temporal loop
for i=1:N_it

    if RK==1
        u=u+dt*compute_rhs(u,r,n_qp_1D,mass,phi_val,phi_grad,phi_val_bd,hx,hy,wts,wts2d,d1,d2,fact_int,fact_bd,complem_fact,radius,pts2d_phi,pts2d_phi_bd,coriolis_fun,eq_type);
    end

    if RK==2
        k1=compute_rhs(u,r,n_qp_1D,mass,phi_val,phi_grad,phi_val_bd,hx,hy,wts,wts2d,d1,d2,fact_int,fact_bd,complem_fact,radius,pts2d_phi,pts2d_phi_bd,coriolis_fun,eq_type);
        k2=compute_rhs(u+dt*k1,r,n_qp_1D,mass,phi_val,phi_grad,phi_val_bd,hx,hy,wts,wts2d,d1,d2,fact_int,fact_bd,complem_fact,radius,pts2d_phi,pts2d_phi_bd,coriolis_fun,eq_type);
        u=u+dt*1/2*k1+dt*1/2*k2;   
    end
    
    if RK==3
        k1=compute_rhs(u,r,n_qp_1D,mass,phi_val,phi_grad,phi_val_bd,hx,hy,wts,wts2d,d1,d2,fact_int,fact_bd,complem_fact,radius,pts2d_phi,pts2d_phi_bd,coriolis_fun,eq_type);
        k2=compute_rhs(u+dt*k1,r,n_qp_1D,mass,phi_val,phi_grad,phi_val_bd,hx,hy,wts,wts2d,d1,d2,fact_int,fact_bd,complem_fact,radius,pts2d_phi,pts2d_phi_bd,coriolis_fun,eq_type);
        k3=compute_rhs(u+dt*(1/4*k1+1/4*k2),r,n_qp_1D,mass,phi_val,phi_grad,phi_val_bd,hx,hy,wts,wts2d,d1,d2,fact_int,fact_bd,complem_fact,radius,pts2d_phi,pts2d_phi_bd,coriolis_fun,eq_type);
        u=u+dt*1/6*k1+dt*1/6*k2+dt*2/3*k3;   
    end
    
    if RK==4
        k1=compute_rhs(u,r,n_qp_1D,mass,phi_val,phi_grad,phi_val_bd,hx,hy,wts,wts2d,d1,d2,fact_int,fact_bd,complem_fact,radius,pts2d_phi,pts2d_phi_bd,coriolis_fun,eq_type);
        k2=compute_rhs(u+dt*k1/2,r,n_qp_1D,mass,phi_val,phi_grad,phi_val_bd,hx,hy,wts,wts2d,d1,d2,fact_int,fact_bd,complem_fact,radius,pts2d_phi,pts2d_phi_bd,coriolis_fun,eq_type);
        k3=compute_rhs(u+dt*(1/2*k2),r,n_qp_1D,mass,phi_val,phi_grad,phi_val_bd,hx,hy,wts,wts2d,d1,d2,fact_int,fact_bd,complem_fact,radius,pts2d_phi,pts2d_phi_bd,coriolis_fun,eq_type);
        k4=compute_rhs(u+dt*(1*k3),r,n_qp_1D,mass,phi_val,phi_grad,phi_val_bd,hx,hy,wts,wts2d,d1,d2,fact_int,fact_bd,complem_fact,radius,pts2d_phi,pts2d_phi_bd,coriolis_fun,eq_type);
        u=u+dt*1/6*k1+dt*1/3*k2+dt*1/3*k3+dt*1/6*k4;   
    end
    
    if RK>=5
        error('RK scheme not implemented');
    end
    
    %plot solution
    if (mod(i-1,100)==0) || i==N_it
        pause(0.2);
        plot_solution(V*u(:,:,1),x_u(:),y_u(:),r,d1,d2,"contour"); 
        fprintf('Iteration %d/%d\n',i,N_it); 
    end
    
    %next iteration
    if t+dt>T
        dt=T-t; 
    end
    t=t+dt;
    
    %avoid useless iterations if solution is already diverging
    if max(abs(u(:)))>=1e8
        fprintf('Iteration %d/%d\n',i,N_it); error('Solution is diverging'); 
    end
    
end

%plot all components of the solution
for i=1:size(u,3)
    figure; plot_solution(V*u(:,:,i),x_u,y_u,r,d1,d2,"surf"); 
end

%compute discretization error, assuming that solution did one complete rotation
if eq_type=="linear" || eq_type=="adv_sphere"
    errL2=0; normL2_sol=0;
    for i=1:d1*d2
        errL2=errL2+(u(:,i,1)-V\u0(:,i,1))'*mass(:,:,i)*(u(:,i,1)-V\u0(:,i,1)); 
    end
    for i=1:d1*d2
        normL2_sol=normL2_sol+(V\u0(:,i,1))'*mass(:,:,i)*(V\u0(:,i,1));
    end
    fprintf('L2 error is %f, and after normalization is %f\n',sqrt(errL2),sqrt(errL2/normL2_sol));
end


fprintf('End of program\n');