clear all; colors=get(gca,'ColorOrder'); close all; clc;
set(0,'defaultAxesFontSize',20); set(0,'defaultLineLineWidth',2);

%definition of the domain [a,b]x[c,d]
%a=0; b=1; c=0; d=1;
%a=0; b=1e7; c=0; d=1e7;
a=0; b=2*pi; c=-pi/2; d=pi/2;

%number of elements in x and y direction
d1=20; d2=20; 

%length of the 1D intervals
hx=(b-a)/d1; hy=(d-c)/d2;

%polynomial degree of DG
r_max=2; 

%cardinality
dim=(r_max+1)^2;

%degree distribution
r=degree_distribution("y_dep",d1,d2,r_max)';  % NEW: r(i,j)

%plot the degree distribution
%%%figure(100);
%%%imagesc(1:d1,1:d2,flipud(r)); colormap jet; colorbar;

%equation type
eq_type="adv_sphere";

%type of quadrature rule (Gauss-Legendre or Gauss-Legendre-Lobatto)
quad_type="leg";

%number of quadrature points in one dimension
n_qp_1D=8;

%number of quadrature points
n_qp=n_qp_1D^2;

%time interval, initial and final time
t=0;
%T=1*(b-a);
%T=100;
%T=36000;
%T=12*86400;
T=5*86400;

%order of the RK scheme (1,2,3,4)
RK=4; 

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
    [unif2d_x{k},unif2d_y{k},~]=tensor_product(unif,unif,nan(size(unif)));
end

%create quadrature points and weights in reference and physical domain,
%both internally and on the boundary of the elements
if quad_type=="leg"
    [pts,wts]=gauss_legendre(n_qp_1D,-1,1); pts=flipud(pts); wts=flipud(wts);
elseif quad_type=="lob"
    [pts,wts]=gauss_legendre_lobatto(n_qp_1D-1); 
end
[pts2d_x,pts2d_y,wts2d]=tensor_product(pts,pts,wts);
 
%Vandermonde matrix, needed to switch from modal to nodal: u_nod=V*u_mod,
%i.e. V(i,j)=Phi_j(x_i) for i,j=1:dim. The x_i are the uniform points.
%Repeat for each degree
V=cell(r_max,1);
for k=1:r_max
    for i=1:(k+1)^2
        for j=1:(k+1)^2
            j_x=floor((j-1)/(k+1))+1;
            j_y=mod(j-1,(k+1))+1;
            V{k}(i,j)=JacobiP(unif2d_x{k}(i),0,0,j_x-1)*JacobiP(unif2d_y{k}(i),0,0,j_y-1);
        end
    end
end

V_1=V{1};  % TODO: calculate explicitly later
V_2=V{2};  % TODO: calculate explicitly later

%Vandermonde matrix, needed to switch from modal to nodal: u_nod=V*u_mod
%only, as this is a rectangular matrix. Repeat for each degree
unif_visual=linspace(-1,1,n_qp_1D)';
[unif2d_visual_x,unif2d_visual_y,~]=tensor_product(unif_visual,unif_visual,nan(size(unif_visual)));
V_rect=cell(r_max,1);
for k=1:r_max
    for i=1:n_qp
        for j=1:(k+1)^2
            j_x=floor((j-1)/(k+1))+1;
            j_y=mod(j-1,(k+1))+1;
            V_rect{k}(i,j)=JacobiP(unif2d_visual_x(i),0,0,j_x-1)*JacobiP(unif2d_visual_y(i),0,0,j_y-1);
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
            phi_val_cell{k}(i,j)=JacobiP(pts2d_x(i),0,0,j_x-1)*JacobiP(pts2d_y(i),0,0,j_y-1);
        end
    end
end
phi_grad_cell_x=cell(r_max,1);
phi_grad_cell_y=cell(r_max,1);
for k=1:r_max
    phi_grad_cell_x{k} = zeros(n_qp,(k+1)^2);
    phi_grad_cell_y{k} = zeros(n_qp,(k+1)^2);
    for i=1:n_qp
        for j=1:(k+1)^2
            j_x=floor((j-1)/(k+1))+1;
            j_y=mod(j-1,(k+1))+1;
            phi_grad_cell_x{k}(i,j)=GradJacobiP(pts2d_x(i),0,0,j_x-1)*JacobiP(pts2d_y(i),0,0,j_y-1);
            phi_grad_cell_y{k}(i,j)=GradJacobiP(pts2d_y(i),0,0,j_y-1)*JacobiP(pts2d_x(i),0,0,j_x-1);
        end
    end
end

%values of basis functions in boundary quadrature points, repeating for
%each face and degree
%dimensions: (num_quad_pts_per_face)x(cardinality)x(num_faces)
phi_val_bd_cell_n=cell(r_max,1);
phi_val_bd_cell_s=cell(r_max,1);
phi_val_bd_cell_e=cell(r_max,1);
phi_val_bd_cell_w=cell(r_max,1);
for k=1:r_max
    phi_val_bd_cell_n{k} = zeros(n_qp_1D,(k+1)^2);
    phi_val_bd_cell_s{k} = zeros(n_qp_1D,(k+1)^2);
    phi_val_bd_cell_e{k} = zeros(n_qp_1D,(k+1)^2);
    phi_val_bd_cell_w{k} = zeros(n_qp_1D,(k+1)^2);
    for i=1:n_qp_1D
        for j=1:(k+1)^2
            j_x=floor((j-1)/(k+1))+1;
            j_y=mod(j-1,(k+1))+1;
            %top
            phi_val_bd_cell_n{k}(i,j)=JacobiP(pts(i),0,0,j_x-1)*JacobiP(1,0,0,j_y-1);
            %bottom
            phi_val_bd_cell_s{k}(i,j)=JacobiP(pts(i),0,0,j_x-1)*JacobiP(-1,0,0,j_y-1);
            %right
            phi_val_bd_cell_e{k}(i,j)=JacobiP(1,0,0,j_x-1)*JacobiP(pts(i),0,0,j_y-1);
            %left
            phi_val_bd_cell_w{k}(i,j)=JacobiP(-1,0,0,j_x-1)*JacobiP(pts(i),0,0,j_y-1);
        end
    end
end

%initial condition (cartesian linear advection) - specified as function
if eq_type=="linear"
    radius = 1.0;  % For completeness -- never used
    h0=0; h1=1; R=(b-a)/2/5; xc=(a+b)/2; yc=(c+d)/2; u0_fun=@(x,y) h0+h1/2*(1+cos(pi*sqrt((x-xc).^2+(y-yc).^2)/R)).*(sqrt((x-xc).^2+(y-yc).^2)<R);
    
    %set initial condition in the uniformly spaced quadrature points
    u0=cell(d1,d2,1);

    for i=1:d1
        for j=1:d2
            local_pos_x = x_c(i) + 0.5*hx*unif2d_x{r(i,j)};   % Vector
            local_pos_y = y_c(j) + 0.5*hy*unif2d_y{r(i,j)};   % Vector
            u0{i,j,1} = u0_fun(local_pos_x,local_pos_y);
        end
    end

elseif eq_type=="swe"

    radius = 1.0;  % For completeness -- never used
%initial condition (cartesian swe) - specified as function
    h0=1000; h1=5; L=1e7; sigma=L/20;
    h0_fun=@(x,y) h0+h1*exp(-((x-L/2).^2+(y-L/2).^2)/(2*sigma^2));
    v0x_fun=@(x,y) zeros(size(x));
    v0y_fun=@(x,y) zeros(size(x));

    u0=cell(d1,d2,3);

    %set initial condition in the uniformly spaced quadrature points
    for i=1:d1
        for j=1:d2
            local_pos_x = x_c(i) + 0.5*hx*unif2d_x{r(i,j)};
            local_pos_y = y_c(j) + 0.5*hy*unif2d_y{r(i,j)};
            u0{i,j,1} = h0_fun(local_pos_x,local_pos_y);
            u0{i,j,2} = h0_fun(local_pos_x,local_pos_y) .* v0x_fun(local_pos_x,local_pos_y);
            u0{i,j,3} = h0_fun(local_pos_x,local_pos_y) .* v0y_fun(local_pos_x,local_pos_y);
        end
    end

    %set coriolis function
    %coriolis_fun=@(x,y) zeros(size(x));
    coriolis_fun=@(x,y) 1e-4*ones(size(x));

%initial condition (spherical linear advection) - specified as function
elseif eq_type=="adv_sphere"

    radius=6.37122e6;
    th_c=pi/2; lam_c=3/2*pi; h0=1000; 
    rr=@(lam,th) radius*acos(sin(th_c)*sin(th)+cos(th_c)*cos(th).*cos(lam-lam_c)); 
    u0_fun=@(lam,th) h0/2*(1+cos(pi*rr(lam,th)/radius)).*(rr(lam,th)<radius/3);
        
    %set initial condition in the uniformly spaced quadrature points
    u0=cell(d1,d2,1);
    for i=1:d1
        for j=1:d2
            local_pos_x = x_c(i) + 0.5*hx*unif2d_x{r(i,j)};
            local_pos_y = y_c(j) + 0.5*hy*unif2d_y{r(i,j)};
            u0{i,j,1} = u0_fun(local_pos_x,local_pos_y);
        end
    end


elseif eq_type=="swe_sphere"

    radius=6.37122e6;
    g=9.80616; h0=2.94e4/g; Omega=7.292e-5; uu0=2*pi*radius/(12*86400); angle=0;
    h0_fun=@(lam,th) h0-1/g*(radius*Omega*uu0+uu0^2/2)*(sin(th)*cos(angle)-cos(lam).*cos(th)*sin(angle)).^2;
    v0x_fun=@(lam,th) uu0.*(cos(th)*cos(angle)+sin(th).*cos(lam)*sin(angle));
    v0y_fun=@(lam,th) -uu0.*sin(angle).*sin(lam);

    %set initial condition in the uniformly spaced quadrature points
    u0=cell(d1,d2,3);
    for i=1:d1
        for j=1:d2
            local_pos_x = x_c(i) + 0.5*hx*unif2d_x{r(i,j)};
            local_pos_y = y_c(j) + 0.5*hy*unif2d_y{r(i,j)};
            u0{i,j,1} = h0_fun(local_pos_x,local_pos_y);
            u0{i,j,2} = h0_fun(local_pos_x,local_pos_y) .* v0x_fun(local_pos_x,local_pos_y);
            u0{i,j,3} = h0_fun(local_pos_x,local_pos_y) .* v0y_fun(local_pos_x,local_pos_y);
        end
    end

end

neq = size(u0,3);   % Number of equations

%if coriolis function is not defined (possibly because not needed), set it
%to zero
if ~exist('coriolis_fun','var')
    coriolis_fun=@(x,y) zeros(size(x));
end

% n2m = nodal2modal(u0,V,r);
u0_check=modal2nodal(nodal2modal(u0,V,r),V,r);

for i=1:d1
    for j=1:d2
        for n=1:neq
            error_2 = norm(u0_check{i,j,n} - u0{i,j,n});
            if error_2 >1e-10
                error('Wrong modal-nodal conversion: %d %d %d error: %e \n',i,j,n,error_2);
            end
        end 
    end
end

%visualize solution at initial time - only first component
x_u=x_e(1:end-1)+(unif_visual+1)*hx/2;
y_u=y_e(1:end-1)+(unif_visual+1)*hy/2;

%%% figure(1); 

u=nodal2modal(u0,V,r); % Only for adv_sphere in this case

to_plot = modal2nodal(u,V_rect,r);
plot_solution(to_plot, x_u(:),y_u(:),n_qp_1D-1,d1,d2,"contour");

[mass, inv_mass]=compute_mass(phi_val_cell,wts2d,d1,d2,r,hx,hy,y_c,pts2d_y,eq_type);

%temporal loop parameters
Courant=dt/min(hx,hy);
N_it=ceil(T/dt);

%print information
fprintf('Space discretization: order %d, elements=%d*%d, domain=[%f,%f]x[%f,%f]\n',r_max,d1,d2,a,b,c,d);
fprintf('Time integration: order %d, T=%f, dt=%f, N_iter=%d\n',RK,T,dt,N_it);

%start temporal loop
for iter=1:N_it

    if RK==1

        rhs_u = compute_rhs(u,r,n_qp_1D,phi_val_cell,phi_grad_cell_x,phi_grad_cell_y,...
                            phi_val_bd_cell_n,phi_val_bd_cell_s,phi_val_bd_cell_e,phi_val_bd_cell_w,inv_mass,...
                            hx,hy,wts,wts2d,radius,pts,pts,pts2d_x,pts2d_y,x_c,y_c,coriolis_fun,eq_type);
        for i=1:d1
            for j=1:d2
                for n=1:neq
                    u{i,j,n} = u{i,j,n}+dt*rhs_u{i,j,n};
                end
            end
        end
    end

    if RK==2

        k1 = compute_rhs(u,r,n_qp_1D,phi_val_cell,phi_grad_cell_x,phi_grad_cell_y,...
                         phi_val_bd_cell_n,phi_val_bd_cell_s,phi_val_bd_cell_e,phi_val_bd_cell_w,inv_mass,...
                         hx,hy,wts,wts2d,radius,pts,pts,pts2d_x,pts2d_y,x_c,y_c,coriolis_fun,eq_type);

% This nasty trick avoids the need for a temporary variable for input to compute_rhs
        for i=1:d1
            for j=1:d2
                for n=1:neq
                    u{i,j,n} = u{i,j,n}+dt*k1{i,j,n};
                end
            end
        end

        k2 = compute_rhs(u,r,n_qp_1D,phi_val_cell,phi_grad_cell_x,phi_grad_cell_y,...
                         phi_val_bd_cell_n,phi_val_bd_cell_s,phi_val_bd_cell_e,phi_val_bd_cell_w,inv_mass,...
                         hx,hy,wts,wts2d,radius,pts,pts,pts2d_x,pts2d_y,x_c,y_c,coriolis_fun,eq_type);

        for i=1:d1
            for j=1:d2
                for n=1:neq
                    u{i,j,n} = u{i,j,n} - dt/2*k1{i,j,n} + dt/2*k2{i,j,n};  % Subtract the unwanted dt/2*k1;
                end
            end
        end


    end

    if RK==3

        k1 = compute_rhs(u,r,n_qp_1D,phi_val_cell,phi_grad_cell_x,phi_grad_cell_y,...
                         phi_val_bd_cell_n,phi_val_bd_cell_s,phi_val_bd_cell_e,phi_val_bd_cell_w,inv_mass,...
                         hx,hy,wts,wts2d,radius,pts,pts,pts2d_x,pts2d_y,x_c,y_c,coriolis_fun,eq_type);

% This nasty trick avoids the need for a temporary variable for input to compute_rhs
        for i=1:d1
            for j=1:d2
                for n=1:neq
                    u{i,j,n} = u{i,j,n}+dt*k1{i,j,n};   % Yields : u+dt*k1
                end
            end
        end

        k2 = compute_rhs(u,r,n_qp_1D,phi_val_cell,phi_grad_cell_x,phi_grad_cell_y,...
                         phi_val_bd_cell_n,phi_val_bd_cell_s,phi_val_bd_cell_e,phi_val_bd_cell_w,inv_mass,...
                         hx,hy,wts,wts2d,radius,pts,pts,pts2d_x,pts2d_y,x_c,y_c,coriolis_fun,eq_type);

        for i=1:d1
            for j=1:d2
                for n=1:neq
                    u{i,j,n} = u{i,j,n}+dt*( -3*k1{i,j,n}/4 + k2{i,j,n}/4);   % Yields u+dt*(1/4*k1 + 1/4*k2)
                end
            end
        end

        k3 = compute_rhs(u,r,n_qp_1D,phi_val_cell,phi_grad_cell_x,phi_grad_cell_y,...
                         phi_val_bd_cell_n,phi_val_bd_cell_s,phi_val_bd_cell_e,phi_val_bd_cell_w,inv_mass,...
                         hx,hy,wts,wts2d,radius,pts,pts,pts2d_x,pts2d_y,x_c,y_c,coriolis_fun,eq_type);

        for i=1:d1
            for j=1:d2
                for n=1:neq
                    u{i,j,n} = u{i,j,n}+dt*( -k1{i,j,n}/12 - k2{i,j,n}/12 + 2*k3{i,j,n}/3); % Yields u+dt*1/6*k1+dt*1/6*k2+dt*2/3*k3;
                end
            end
        end
        
    end
    
    if RK==4

        k1 = compute_rhs(u,r,n_qp_1D,phi_val_cell,phi_grad_cell_x,phi_grad_cell_y,...
                         phi_val_bd_cell_n,phi_val_bd_cell_s,phi_val_bd_cell_e,phi_val_bd_cell_w,inv_mass,...
                         hx,hy,wts,wts2d,radius,pts,pts,pts2d_x,pts2d_y,x_c,y_c,coriolis_fun,eq_type);

% This nasty trick avoids the need for a temporary variable for input to compute_rhs
        for i=1:d1
            for j=1:d2
                for n=1:neq
                    u{i,j,n} = u{i,j,n}+dt*k1{i,j,n}/2;   % Yields : u+dt*k1/2
                end
            end
        end

        k2 = compute_rhs(u,r,n_qp_1D,phi_val_cell,phi_grad_cell_x,phi_grad_cell_y,...
                         phi_val_bd_cell_n,phi_val_bd_cell_s,phi_val_bd_cell_e,phi_val_bd_cell_w,inv_mass,...
                         hx,hy,wts,wts2d,radius,pts,pts,pts2d_x,pts2d_y,x_c,y_c,coriolis_fun,eq_type);

        for i=1:d1
            for j=1:d2
                for n=1:neq
                    u{i,j,n} = u{i,j,n}+dt*(-k1{i,j,n}/2+k2{i,j,n}/2);   % Yields : u+dt*(1/2*k2)
                end
            end
        end

        k3 = compute_rhs(u,r,n_qp_1D,phi_val_cell,phi_grad_cell_x,phi_grad_cell_y,...
                         phi_val_bd_cell_n,phi_val_bd_cell_s,phi_val_bd_cell_e,phi_val_bd_cell_w,inv_mass,...
                         hx,hy,wts,wts2d,radius,pts,pts,pts2d_x,pts2d_y,x_c,y_c,coriolis_fun,eq_type);

        for i=1:d1
            for j=1:d2
                for n=1:neq
                    u{i,j,n} = u{i,j,n}+dt*(-k2{i,j,n}/2 +k3{i,j,n});   % Yields : u+dt*(1*k3)
                end
            end
        end

        k4 = compute_rhs(u,r,n_qp_1D,phi_val_cell,phi_grad_cell_x,phi_grad_cell_y,...
                         phi_val_bd_cell_n,phi_val_bd_cell_s,phi_val_bd_cell_e,phi_val_bd_cell_w,inv_mass,...
                         hx,hy,wts,wts2d,radius,pts,pts,pts2d_x,pts2d_y,x_c,y_c,coriolis_fun,eq_type);

        for i=1:d1
            for j=1:d2
                for n=1:neq
                    u{i,j,n} = u{i,j,n}+dt*(k1{i,j,n}/6+k2{i,j,n}/3-2*k3{i,j,n}/3+k4{i,j,n}/6);   
% Yields final result:  u=u+dt*1/6*k1+dt*1/3*k2+dt*1/3*k3+dt*1/6*k4
                end
            end
        end

    end
    
    if RK>=5
        error('RK scheme higher than 4 not implemented');
    end
    
%plot solution
    if (mod(iter-1,plot_freq)==0) || iter==N_it
        fprintf('Iteration %d/%d\n',iter,N_it); 
        figure(1);
        pause(0.05);
        to_plot = modal2nodal(u,V_rect,r);
        plot_solution(to_plot, x_u(:),y_u(:),n_qp_1D-1,d1,d2,"contour");
    end

%next iteration
    if t+dt>T
        dt=T-t; 
    end
    t=t+dt;
%%%    
%%%    %avoid useless iterations if solution is already diverging
%%%    if max(abs(u(:)))>=1e8
%%%        fprintf('Iteration %d/%d\n',i,N_it); error('Solution is diverging'); 
%%%    end
%%%    

end

fprintf('End of program\n');