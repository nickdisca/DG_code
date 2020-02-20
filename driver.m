clear all; colors=get(gca,'ColorOrder'); close all; clc;
set(0,'defaultAxesFontSize',20); set(0,'defaultLineLineWidth',2);

%definition of the domain [a,b]x[c,d]
%a=0; b=1; c=0; d=1;
%a=0; b=1e7; c=0; d=1e7;
a=0; b=2*pi; c=-pi/2; d=pi/2;

%number of elements in x and y direction
d1=7; d2=5; 

%length of the 1D intervals
hx=(b-a)/d1; hy=(d-c)/d2;

%polynomial degree of DG
r_max=2; 

%cardinality
dim=(r_max+1)^2;

%degree distribution
r=degree_distribution("y_dep",d1,d2,r_max);
r_new = r';  % r_new(i,j) instead of r(j,i)

%dynamic adaptivity
dyn_adapt=false;

%plot the degree distribution
figure(100);
imagesc(1:d1,1:d2,flipud(r)); colormap jet; colorbar;

%equation type
eq_type="adv_sphere";

%type of quadrature rule (Gauss-Legendre or Gauss-Legendre-Lobatto)
quad_type="leg";

%number of quadrature points in one dimension
n_qp_1D=4;

%number of quadrature points
n_qp=n_qp_1D^2;

%time interval, initial and final time
t=0;
%T=1*(b-a);
T=36000;
%T=12*86400;
%T=5*86400;

%order of the RK scheme (1,2,3,4)
RK=3; 

%time step
%dt=1/r_max^2*min(hx,hy)*0.1; 
dt=100;
%dt=50;

%plotting frequency
plot_freq=100;

%beginning and end of the intervals
x_e=linspace(a,b,d1+1); 
y_e=linspace(c,d,d2+1);

half_cell_x = (b-a)/d1/2;
half_cell_y = (d-c)/d2/2;
x_c=linspace(a+half_cell_x,b-half_cell_x,d1); % Cell centers in X
y_c=linspace(c+half_cell_y,d-half_cell_y,d2); % Cell centers in Y

%create uniformly spaced points in reference and physical domain, and
%repeat for each degree
unif2d=cell(r_max,1);
unif2d_x=cell(r_max);
unif2d_y=cell(r_max);
for k=1:r_max
    unif=linspace(-1,1,k+1)';
    [unif2d{k},~]=tensor_product(unif,unif,nan(size(unif)));
    unif2d_x{k} = unif2d{k}(:,1);
    unif2d_y{k} = unif2d{k}(:,2);
end

%map to physical domain
unif2d_phi=map2phi_adaptive(unif2d,r,r_max,x_e,y_e,d1,d2,hx,hy);

%create quadrature points and weights in reference and physical domain,
%both internally and on the boundary of the elements
if quad_type=="leg"
    [pts,wts]=gauss_legendre(n_qp_1D,-1,1); pts=flipud(pts); wts=flipud(wts);
elseif quad_type=="lob"
    [pts,wts]=gauss_legendre_lobatto(n_qp_1D-1); 
end
[pts2d,wts2d]=tensor_product(pts,pts,wts);
pts2d_x = pts2d(:,1);
pts2d_y = pts2d(:,2);
pts2d_phi=map2phi_static(pts2d,n_qp_1D-1,x_e,y_e,d1,d2,hx,hy);
pts2d_phi_bd=map2phi_bd(pts,n_qp_1D-1,x_e,y_e,d1,d2,hx,hy);

%Vandermonde matrix, needed to switch from modal to nodal: u_nod=V*u_mod,
%i.e. V(i,j)=Phi_j(x_i) for i,j=1:dim. The x_i are the uniform points.
%Repeat for each degree
V=cell(r_max,1);
for k=1:r_max
    for i=1:(k+1)^2
        for j=1:(k+1)^2
            j_x=floor((j-1)/(k+1))+1;
            j_y=mod(j-1,(k+1))+1;
            V{k}(i,j)=JacobiP(unif2d{k}(i,1),0,0,j_x-1)*JacobiP(unif2d{k}(i,2),0,0,j_y-1);
        end
    end
end

V_1=V{1};  % TODO: calculate explicitly later
V_2=V{2};  % TODO: calculate explicitly later

%Vandermonde matrix, needed to switch from modal to nodal: u_nod=V*u_mod
%only, as this is a rectangular matrix. Repeat for each degree
unif_visual=linspace(-1,1,n_qp_1D)';
[unif2d_visual,~]=tensor_product(unif_visual,unif_visual,nan(size(unif_visual)));
V_rect=cell(r_max,1);
for k=1:r_max
    for i=1:n_qp
        for j=1:(k+1)^2
            j_x=floor((j-1)/(k+1))+1;
            j_y=mod(j-1,(k+1))+1;
            V_rect{k}(i,j)=JacobiP(unif2d_visual(i,1),0,0,j_x-1)*JacobiP(unif2d_visual(i,2),0,0,j_y-1);
        end
    end
end

V_rect_1 = V_rect{1};
V_rect_2 = V_rect{2};

%values and grads of basis functions in internal quadrature points, i.e.
%phi_val(i,j)=Phi_j(x_i) for i=1:dim_qp,j=1:dim. The x_i are the quadrature points,
%the gradient has two components due to the x and y derivatives. Repeat for each degree
phi_val_cell=cell(r_max,1);
for k=1:r_max
    for i=1:n_qp
        for j=1:(k+1)^2
            j_x=floor((j-1)/(k+1))+1;
            j_y=mod(j-1,(k+1))+1;
            phi_val_cell{k}(i,j)=JacobiP(pts2d(i,1),0,0,j_x-1)*JacobiP(pts2d(i,2),0,0,j_y-1);
        end
    end
end
phi_grad_cell=cell(r_max,1);
for k=1:r_max
    for i=1:n_qp
        for j=1:(k+1)^2
            j_x=floor((j-1)/(k+1))+1;
            j_y=mod(j-1,(k+1))+1;
            phi_grad_cell{k}(i,j,1)=GradJacobiP(pts2d(i,1),0,0,j_x-1)*JacobiP(pts2d(i,2),0,0,j_y-1);
            phi_grad_cell{k}(i,j,2)=GradJacobiP(pts2d(i,2),0,0,j_y-1)*JacobiP(pts2d(i,1),0,0,j_x-1);
        end
    end
end

%values of basis functions in boundary quadrature points, repeating for
%each face and degree
%dimensions: (num_quad_pts_per_face)x(cardinality)x(num_faces)
phi_val_bd_cell=cell(r_max,1);
for k=1:r_max
    for i=1:n_qp_1D
        for j=1:(k+1)^2
            j_x=floor((j-1)/(k+1))+1;
            j_y=mod(j-1,(k+1))+1;
            %right
            phi_val_bd_cell{k}(i,j,2)=JacobiP(1,0,0,j_x-1)*JacobiP(pts(i),0,0,j_y-1);
            %left
            phi_val_bd_cell{k}(i,j,4)=JacobiP(-1,0,0,j_x-1)*JacobiP(pts(i),0,0,j_y-1);
            %bottom
            phi_val_bd_cell{k}(i,j,1)=JacobiP(pts(i),0,0,j_x-1)*JacobiP(-1,0,0,j_y-1);
            %top
            phi_val_bd_cell{k}(i,j,3)=JacobiP(pts(i),0,0,j_x-1)*JacobiP(1,0,0,j_y-1);
        end
    end
end

%compute factor (cosine for sphere, 1 for cartesian)
[fact_int,fact_bd,complem_fact,radius]=compute_factor(eq_type,pts2d_phi(n_qp+1:2*n_qp,:),pts2d_phi_bd(n_qp_1D+1:2*n_qp_1D,:,:));

%initial condition (cartesian linear advection) - specified as function
if eq_type=="linear"
    %u0_fun=@(x,y) sin(2*pi*x).*sin(2*pi*y);
    h0=0; h1=1; R=(b-a)/2/5; x_c=(a+b)/2; y_c=(c+d)/2; u0_fun=@(x,y) h0+h1/2*(1+cos(pi*sqrt((x-x_c).^2+(y-y_c).^2)/R)).*(sqrt((x-x_c).^2+(y-y_c).^2)<R);
    %u0_fun=@(x,y) 5*(~isnan(x));
    
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
    %coriolis_fun=@(x,y) zeros(size(x));
    coriolis_fun=@(x,y) 1e-4*ones(size(x));
    
%initial condition (spherical linear advection) - specified as function
elseif eq_type=="adv_sphere"
    th_c=pi/2; lam_c=3/2*pi; h0=1000; 
    rr=@(lam,th) radius*acos(sin(th_c)*sin(th)+cos(th_c)*cos(th).*cos(lam-lam_c)); 
    u0_fun=@(lam,th) h0/2*(1+cos(pi*rr(lam,th)/radius)).*(rr(lam,th)<radius/3);
        
    %set initial condition in the uniformly spaced quadrature points
    u0=u0_fun(unif2d_phi(1:dim,:),unif2d_phi(dim+1:2*dim,:));  
    u0_new=cell(d1,d2);
    for i=1:d1
        for j=1:d2
            local_pos_x = x_c(i) + unif2d_x{r_new(i,j)}/(2*pi)/d1;
            local_pos_y = y_c(j) + unif2d_y{r_new(i,j)}/pi/2/d2;
            u0_new{i,j} = u0_fun(local_pos_x,local_pos_y);
        end
    end
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

%do modal-nodal conversion check
u0_check=modal2nodal(nodal2modal(u0(:,:,1),V,r),V,r);
max_error = max(max(abs(u0_check-u0(:,:,1))));
if max_error>1e-10
    error('Wrong modal-nodal conversion: error %e \n',max_error);
end

% n2m = nodal2modal_new(u0_new,V,r_new);
u0_check_new=modal2nodal_new(nodal2modal_new(u0_new,V,r_new),V,r_new);

for i=1:d1
    for j=1:d2
        error_2 = norm(u0_check_new{i,j} - u0_new{i,j});
        if error_2 >1e-10
            error('Wrong modal-nodal conversion: %d %d error: %e \n',i,j,error_2);
        end 
    end
end

%visualize solution at initial time - only first component
x_u=x_e(1:end-1)+(unif_visual+1)/2*hx;
y_u=y_e(1:end-1)+(unif_visual+1)/2*hy;

figure(1); 
%plot_solution( modal2nodal(nodal2modal(u0(:,:,1),V,r),V_rect,r) ,x_u(:),y_u(:),n_qp_1D-1,d1,d2,"contour");

to_plot = modal2nodal_new(nodal2modal_new(u0_new,V,r_new),V_rect,r_new)
plot_solution_new(to_plot, x_u(:),y_u(:),n_qp_1D-1,d1,d2,"contour");
%convert nodal to modal: the vector u will contain the modal coefficient
u=zeros(dim,d1*d2,size(u0,3)); 
for i=1:size(u0,3)
    u(:,:,i)=nodal2modal(u0(:,:,i),V,r); 
end

u_new=nodal2modal_new(u0_new,V,r_new); % Only for adv_sphere in this case

%compute mass matrix and its inverse
[mass_tensor, inv_mass_tensor]=compute_mass(phi_val_cell,wts2d,d1,d2,r,hx,hy,fact_int);

%compute mass matrix and its inverse
%[mass_tensor_new, inv_mass_tensor_new]=compute_mass_new(phi_val_cell,wts2d,d1,d2,r_new,hx,hy,fact_int_new);

%convert to global matrices instead of tensors
idx_r=repelem(reshape((1:dim*d1*d2),dim,d1*d2),1,dim);
idx_c=repmat(1:dim*d1*d2,dim,1);
mass=sparse(idx_r(:),idx_c(:),mass_tensor(:));
inv_mass=sparse(idx_r(:),idx_c(:),inv_mass_tensor(:));

%convert to global matrices instead of cell array
phi_val=convert_cell_2_global(phi_val_cell,r,n_qp,dim,d1,d2);
phi_grad=convert_cell_2_global(phi_grad_cell,r,n_qp,dim,d1,d2);
phi_val_bd=convert_cell_2_global(phi_val_bd_cell,r,n_qp_1D,dim,d1,d2);


%temporal loop parameters
Courant=dt/min(hx,hy);
N_it=ceil(T/dt);

%print information
fprintf('Space discretization: order %d, elements=%d*%d, domain=[%f,%f]x[%f,%f]\n',r_max,d1,d2,a,b,c,d);
fprintf('Time integration: order %d, T=%f, dt=%f, N_iter=%d\n',RK,T,dt,N_it);

%start temporal loop
for iter=1:N_it
    
    if dyn_adapt
        error('Dynamic adpativity is not supported');
    end

    if RK==1
        u=u+dt*compute_rhs(u,r,n_qp_1D,mass,inv_mass,phi_val,phi_grad,phi_val_bd,hx,hy,wts,wts2d,d1,d2,fact_int,fact_bd,complem_fact,radius,pts2d_phi,pts2d_phi_bd,coriolis_fun,eq_type);
    end

    if RK==2
        k1=compute_rhs(u,r,n_qp_1D,mass,inv_mass,phi_val,phi_grad,phi_val_bd,hx,hy,wts,wts2d,d1,d2,fact_int,fact_bd,complem_fact,radius,pts2d_phi,pts2d_phi_bd,coriolis_fun,eq_type);
        k2=compute_rhs(u+dt*k1,r,n_qp_1D,mass,inv_mass,phi_val,phi_grad,phi_val_bd,hx,hy,wts,wts2d,d1,d2,fact_int,fact_bd,complem_fact,radius,pts2d_phi,pts2d_phi_bd,coriolis_fun,eq_type);
        u=u+dt*1/2*k1+dt*1/2*k2;   
    end
    
    if RK==3
        k1=compute_rhs(u,r,n_qp_1D,mass,inv_mass,phi_val,phi_grad,phi_val_bd,hx,hy,wts,wts2d,d1,d2,fact_int,fact_bd,complem_fact,radius,pts2d_phi,pts2d_phi_bd,coriolis_fun,eq_type);
        k2=compute_rhs(u+dt*k1,r,n_qp_1D,mass,inv_mass,phi_val,phi_grad,phi_val_bd,hx,hy,wts,wts2d,d1,d2,fact_int,fact_bd,complem_fact,radius,pts2d_phi,pts2d_phi_bd,coriolis_fun,eq_type);
        k3=compute_rhs(u+dt*(1/4*k1+1/4*k2),r,n_qp_1D,mass,inv_mass,phi_val,phi_grad,phi_val_bd,hx,hy,wts,wts2d,d1,d2,fact_int,fact_bd,complem_fact,radius,pts2d_phi,pts2d_phi_bd,coriolis_fun,eq_type);
        u=u+dt*1/6*k1+dt*1/6*k2+dt*2/3*k3;   
    end
    
    if RK==4
        k1=compute_rhs(u,r,n_qp_1D,mass,inv_mass,phi_val,phi_grad,phi_val_bd,hx,hy,wts,wts2d,d1,d2,fact_int,fact_bd,complem_fact,radius,pts2d_phi,pts2d_phi_bd,coriolis_fun,eq_type);
        k2=compute_rhs(u+dt*k1/2,r,n_qp_1D,mass,inv_mass,phi_val,phi_grad,phi_val_bd,hx,hy,wts,wts2d,d1,d2,fact_int,fact_bd,complem_fact,radius,pts2d_phi,pts2d_phi_bd,coriolis_fun,eq_type);
        k3=compute_rhs(u+dt*(1/2*k2),r,n_qp_1D,mass,inv_mass,phi_val,phi_grad,phi_val_bd,hx,hy,wts,wts2d,d1,d2,fact_int,fact_bd,complem_fact,radius,pts2d_phi,pts2d_phi_bd,coriolis_fun,eq_type);
        k4=compute_rhs(u+dt*(1*k3),r,n_qp_1D,mass,inv_mass,phi_val,phi_grad,phi_val_bd,hx,hy,wts,wts2d,d1,d2,fact_int,fact_bd,complem_fact,radius,pts2d_phi,pts2d_phi_bd,coriolis_fun,eq_type);
        u=u+dt*1/6*k1+dt*1/3*k2+dt*1/3*k3+dt*1/6*k4;   
    end
    
    if RK>=5
        error('RK scheme not implemented');
    end
    
    %plot solution
    if (mod(iter-1,plot_freq)==0) || iter==N_it
        fprintf('Iteration %d/%d\n',iter,N_it); 
        figure(1);
        pause(0.05);
        plot_solution( modal2nodal(u,V_rect,r) ,x_u(:),y_u(:),n_qp_1D-1,d1,d2,"contour");
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
%    figure(200); 
%    plot_solution(modal2nodal(u(:,:,i),V_rect,r),x_u,y_u,n_qp_1D-1,d1,d2,"surf"); 
end

%compute discretization error, assuming that solution did one complete rotation
if eq_type=="linear" || eq_type=="adv_sphere"
    errL2=reshape(u(:,:,1)-nodal2modal(u0(:,:,1),V,r),dim*d1*d2,1)'*mass*reshape(u(:,:,1)-nodal2modal(u0(:,:,1),V,r),dim*d1*d2,1);
    normL2_sol=reshape(nodal2modal(u0(:,:,1),V,r),dim*d1*d2,1)'*mass*reshape(nodal2modal(u0(:,:,1),V,r),dim*d1*d2,1);
    fprintf('L2 error is %f, and after normalization is %f\n',sqrt(errL2),sqrt(errL2/normL2_sol));
end


fprintf('End of program\n');