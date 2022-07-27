clear all; colors=get(gca,'ColorOrder'); close all;
set(0,'defaultAxesFontSize',20); set(0,'defaultLineLineWidth',2);

%equation type
% eq_type="linear";
eq_type="adv_sphere";
% eq_type="swe";
% eq_type="swe_sphere";

% plot_type = "surf";
plot_type = "contour";
% plot_type = "sphere";



debug = 0;
debug_freq=100;

%definition of the domain [a,b]x[c,d]
if eq_type == "linear"
    a=0; b=1; c=0; d=1;
elseif eq_type == "swe"
    a=0; b=1e7; c=0; d=1e7;
elseif eq_type == "adv_sphere"
%     a=0; b=1e7; c=0; d=1e7;
    a=0; b=2*pi; c=-pi/2; d=pi/2;
elseif eq_type == "swe_sphere"
    a=0; b=2*pi; c=-pi/2; d=pi/2;
end

%number of elements in x and y direction
d1=20; d2=20; 

%length of the 1D intervals
hx=(b-a)/d1; hy=(d-c)/d2;

%polynomial degree of DG
r_max=3; 
%order of the RK scheme (1,2,3,4)
RK=1; 

%cardinality
dim=(r_max+1)^2;

%degree distribution
r=degree_distribution("unif",d1,d2,r_max);

%dynamic adaptivity
dyn_adapt=false;

%plot the degree distribution
% figure(1);
% imagesc(1:d1,1:d2,flipud(r)); colormap jet; colorbar;

%type of quadrature rule (Gauss-Legendre or Gauss-Legendre-Lobatto)
quad_type="leg";

%number of quadrature points in one dimension
n_qp_1D=r_max+1;

%number of quadrature points
n_qp=n_qp_1D^2;

%time interval, initial and final time
t=0;
if eq_type == "linear"
    T = 1;
elseif eq_type == "adv_sphere"
    T=12*86400;
elseif eq_type == "swe_sphere"
    T=5*86400;  % SWE SPHERE CASE 2
else
    T=12*86400;
end


%time step

if eq_type == "linear"
    dt=1/r_max^2*min(hx,hy)*0.1;
elseif eq_type == "adv_sphere"
    if r_max == 1
        dt=500; 
    else
        dt=50;
    end
    T=12*86400;

elseif eq_type == "swe"
    dt=1/r_max^2*min(hx,hy)*0.0005;
    Courant = 0.0001;
elseif eq_type == "swe_sphere"
  dt = 3.39;  
end

%beginning and end of the intervals
x_e=linspace(a,b,d1+1); 
y_e=linspace(c,d,d2+1);

%create uniformly spaced points in reference and physical domain, and
%repeat for each degree
unif2d=cell(r_max,1);
for k=1:r_max
    unif=linspace(-1,1,k+1)';
    [unif2d{k},~]=tensor_product(unif,unif,nan(size(unif)));
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
% TODO identical for all x values
[fact_int,fact_bd,complem_fact,radius]=compute_factor(eq_type,pts2d_phi(n_qp+1:2*n_qp,:),pts2d_phi_bd(n_qp_1D+1:2*n_qp_1D,:,:));

%initial condition (cartesian linear advection) - specified as function
if eq_type=="linear"
    % Sine Wave
    u0_fun=@(x,y) sin(2*pi*x).*sin(2*pi*y); 
    
    % Williamson Test
%     h0=0; h1=1; R=(b-a)/2/5; x_c=(a+b)/2; y_c=(c+d)/2;
%     u0_fun=@(x,y) h0+h1/2*(1+cos(pi*sqrt((x-x_c).^2+(y-y_c).^2)/R)).*(sqrt((x-x_c).^2+(y-y_c).^2)<R);
    %u0_fun=@(x,y) 5*(~isnan(x));



    
    %set initial condition in the uniformly spaced quadrature points
    u0=u0_fun(unif2d_phi(1:dim,:),unif2d_phi(dim+1:2*dim,:));
    
%     u0 = ones(dim, d1*d2);

%initial condition (cartesian swe) - specified as function
elseif eq_type=="swe"
%     h0=1000; h1=5; L=1e7; sigma=L/20;
%     h0_fun=@(x,y) h0+h1*exp(-((x-L/2).^2+(y-L/2).^2)/(2*sigma^2));
    h0=1000; h1=100; Cx=a+(b-a)*1/2; Cy=c+(d-c)*1/2; sigma=(a+b)/20;
    speed = sqrt((h0+h1)*9.81);
    fprintf("Expected speed: %f; Reaches boundary in %f\n", speed, 0.5/speed);
    h0_fun=@(x,y) h0+h1*exp(-((x-Cx).^2+(y-Cy).^2)/(2*sigma^2));

    v0x_fun=@(x,y) zeros(size(x));
    v0y_fun=@(x,y) zeros(size(x));
    
    %set initial condition in the uniformly spaced quadrature points
    u0(:,:,1)=h0_fun(unif2d_phi(1:dim,:),unif2d_phi(dim+1:2*dim,:));
    u0(:,:,2)=h0_fun(unif2d_phi(1:dim,:),unif2d_phi(dim+1:2*dim,:)).*v0x_fun(unif2d_phi(1:dim,:),unif2d_phi(dim+1:2*dim,:));
    u0(:,:,3)=h0_fun(unif2d_phi(1:dim,:),unif2d_phi(dim+1:2*dim,:)).*v0y_fun(unif2d_phi(1:dim,:),unif2d_phi(dim+1:2*dim,:));
    
    %set coriolis function
    %coriolis_fun=@(x,y) zeros(size(x));
%     coriolis_fun=@(x,y) 1e-4*ones(size(x));
    
%initial condition (spherical linear advection) - specified as function
elseif eq_type=="adv_sphere"
    th_c=0; lam_c=3/2*pi; h0=1000; 

%     th_c=0; lam_c=pi/2; h0=1000000; 
    rr=@(lam,th) radius*acos(sin(th_c)*sin(th)+cos(th_c)*cos(th).*cos(lam-lam_c)); 
    u0_fun=@(lam,th) h0/2*(1+cos(pi*rr(lam,th)/radius)).*(rr(lam,th)<radius/3);
    
    %set initial condition in the uniformly spaced quadrature points
    u0=u0_fun(unif2d_phi(1:dim,:),unif2d_phi(dim+1:2*dim,:));  
    
%initial condition (spherical swe) - specified as function
elseif eq_type=="swe_sphere"
    
    % === Case 2 Williamson ===
%     g=9.80616; h0=2.94e4/g; Omega=7.292e-5; uu0=2*pi*radius/(12*86400); angle=0;
%     h0_fun=@(lam,th) h0-1/g*(radius*Omega*uu0+uu0^2/2)*(sin(th)*cos(angle)-cos(lam).*cos(th)*sin(angle)).^2;
%     v0x_fun=@(lam,th) uu0.*(cos(th)*cos(angle)+sin(th).*cos(lam)*sin(angle));
%     v0y_fun=@(lam,th) -uu0.*sin(angle).*sin(lam);
%     
% %     set initial condition in the uniformly spaced quadrature points
%     u0(:,:,1)=h0_fun(unif2d_phi(1:dim,:),unif2d_phi(dim+1:2*dim,:));
%     u0(:,:,2)=h0_fun(unif2d_phi(1:dim,:),unif2d_phi(dim+1:2*dim,:)).*v0x_fun(unif2d_phi(1:dim,:),unif2d_phi(dim+1:2*dim,:));
%     u0(:,:,3)=h0_fun(unif2d_phi(1:dim,:),unif2d_phi(dim+1:2*dim,:)).*v0y_fun(unif2d_phi(1:dim,:),unif2d_phi(dim+1:2*dim,:));
%     
% %     set coriolis function
% %     coriolis_fun=@(x,y) zeros(size(x));
%     coriolis_fun=@(lam,th) 2*Omega*(sin(th)*cos(angle)-cos(th).*cos(lam)*sin(angle));

    % === Case 6 ===
    g=9.80616; Omega=7.292e-5; omega=7.848e-6; K=omega; h0=8e3; R=4;
    A_fun=@(lam,th) 0.5*omega * (2*Omega+omega) * cos(th).^2 + 0.25*K*K*cos(th).^(2*R) .* ((R+1)*cos(th).^2 + (2*R*R - R - 2) - 2*R*R*cos(th).^(-2));
    B_fun=@(lam,th) 2*(Omega+omega)*K*cos(th).^R .* ((R*R + 2*R + 2) - (R+1)^2*cos(th).^2) / ((R+1) * (R+2));
    C_fun=@(lam,th) 0.25*K*K*cos(th).^(2*R) .* ((R+1)*cos(th).^2 - (R+2));
    
    h0_fun=@(lam,th) h0 + radius*radius/g*(A_fun(lam,th) + B_fun(lam,th).*cos(R*lam) + C_fun(lam,th).*cos(2*R*lam));
    v0x_fun=@(lam,th) radius*omega*cos(th) + radius*K*cos(th).^(R-1) .* (R*sin(th).^2 - cos(th).^2) .* (cos(R*lam));
    v0y_fun=@(lam,th) -radius*K*R*cos(th).^(R-1).*sin(th).*sin(R*lam);

    
    %set initial condition in the uniformly spaced quadrature points
    u0(:,:,1)=h0_fun(unif2d_phi(1:dim,:),unif2d_phi(dim+1:2*dim,:));
    u0(:,:,2)=h0_fun(unif2d_phi(1:dim,:),unif2d_phi(dim+1:2*dim,:)).*v0x_fun(unif2d_phi(1:dim,:),unif2d_phi(dim+1:2*dim,:));
    u0(:,:,3)=h0_fun(unif2d_phi(1:dim,:),unif2d_phi(dim+1:2*dim,:)).*v0y_fun(unif2d_phi(1:dim,:),unif2d_phi(dim+1:2*dim,:));
    
%     set coriolis function
    coriolis_fun=@(lam,th) 2*Omega*sin(th);
    
    alpha = max( sqrt(abs(g*u0(:,:,1)))+sqrt((u0(:,:,2)./u0(:,:,1)).^2+(u0(:,:,3)./u0(:,:,1)).^2),[],'all')
end

%if coriolis function is not defined (possibly because not needed), set it
%to zero
if ~exist('coriolis_fun','var')
    coriolis_fun=@(x,y) zeros(size(x));
end

%do modal-nodal conversion check
% u0_check=modal2nodal(nodal2modal(u0(:,:,1),V,r),V,r);
% if max(max(abs(u0_check-u0(:,:,1))))>1e-10
%     error('Wrong modal-nodal conversion: %e\n',max(max(abs(u0_check-u0(:,:,1)))));
% end

%visualize solution at initial time - only first component
x_u=x_e(1:end-1)+(unif_visual+1)/2*hx;
y_u=y_e(1:end-1)+(unif_visual+1)/2*hy;

% PLOT INITIAL CONDITIONS
% f1 = figure('units','normalized','outerposition',[0 0 1 1]);  
% figure(f1);
% uN_0 = modal2nodal(nodal2modal(u0(:,:,:),V,r),V_rect,r);
% if eq_type == "swe" || eq_type == "swe_sphere"
%     uN_0(:,:,2:3) = uN_0(:,:,2:3) ./ uN_0(:,:,1);
% end
% plot_solution(uN_0,x_u(:),y_u(:),n_qp_1D-1,d1,d2,radius,plot_type, "Intial Conditions", eq_type, fact_int);
% uN_0

u = nodal2modal(u0, V, r);

%compute mass matrix and its inverse
[mass_tensor, inv_mass_tensor]=compute_mass(phi_val_cell,wts2d,d1,d2,r,hx,hy,fact_int);

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
N_it=ceil(T/dt);
%plotting frequency
if eq_type == "linear"
    plot_freq = N_it / 10;
else
    plot_freq=100;
end

% courant = 0.009;
% dt = courant * min(radius * sin(hx) * sin(hy), radius * sin(hx) * cos(hy)) / ((r_max+1) * alpha)


if eq_type == "swe_sphere"
    h_max = max(u0(:,:,1), [], 'all');
    u_max = max(u0(:,:,2)./u0(:,:,1), [], 'all');
    v_max = max(u0(:,:,3)./u0(:,:,1), [], 'all');
    
%     dt = Courant * min(hx, hy) / (r_max * (max(u_max, v_max) + sqrt(g*h_max)));

end
% return

%print information
fprintf('Space discretization: order %d, elements=%d*%d, domain=[%f,%f]x[%f,%f]\n',r_max,d1,d2,a,b,c,d);
fprintf('Time integration: order %d, T=%f, dt=%f, N_iter=%d\n\n',RK,T,dt,N_it);

f2 = figure('units','normalized','outerposition',[0 0 1 1]);  
figure(f2);
%start temporal loop
tic;
for iter=1:N_it
    
    if dyn_adapt
        error('Dynamic adpativity is not supported');
    end

    if RK==1
        u=u+dt*compute_rhs(u,r,n_qp_1D,mass,inv_mass,phi_val,phi_grad,phi_val_bd,hx,hy,wts,wts2d,d1,d2,fact_int,fact_bd,complem_fact,radius,pts2d_phi,pts2d_phi_bd,coriolis_fun,eq_type);

    elseif RK==2
        k1=compute_rhs(u,r,n_qp_1D,mass,inv_mass,phi_val,phi_grad,phi_val_bd,hx,hy,wts,wts2d,d1,d2,fact_int,fact_bd,complem_fact,radius,pts2d_phi,pts2d_phi_bd,coriolis_fun,eq_type);
        k2=compute_rhs(u+dt*k1,r,n_qp_1D,mass,inv_mass,phi_val,phi_grad,phi_val_bd,hx,hy,wts,wts2d,d1,d2,fact_int,fact_bd,complem_fact,radius,pts2d_phi,pts2d_phi_bd,coriolis_fun,eq_type);
        u=u+dt*1/2*k1+dt*1/2*k2;   
    
    elseif RK==3
        k1=compute_rhs(u,r,n_qp_1D,mass,inv_mass,phi_val,phi_grad,phi_val_bd,hx,hy,wts,wts2d,d1,d2,fact_int,fact_bd,complem_fact,radius,pts2d_phi,pts2d_phi_bd,coriolis_fun,eq_type);
        k2=compute_rhs(u+dt*k1,r,n_qp_1D,mass,inv_mass,phi_val,phi_grad,phi_val_bd,hx,hy,wts,wts2d,d1,d2,fact_int,fact_bd,complem_fact,radius,pts2d_phi,pts2d_phi_bd,coriolis_fun,eq_type);
        k3=compute_rhs(u+dt*(1/4*k1+1/4*k2),r,n_qp_1D,mass,inv_mass,phi_val,phi_grad,phi_val_bd,hx,hy,wts,wts2d,d1,d2,fact_int,fact_bd,complem_fact,radius,pts2d_phi,pts2d_phi_bd,coriolis_fun,eq_type);
        u=u+dt*1/6*k1+dt*1/6*k2+dt*2/3*k3;   
    
    elseif RK==4
        k1=compute_rhs(u,r,n_qp_1D,mass,inv_mass,phi_val,phi_grad,phi_val_bd,hx,hy,wts,wts2d,d1,d2,fact_int,fact_bd,complem_fact,radius,pts2d_phi,pts2d_phi_bd,coriolis_fun,eq_type);
        k2=compute_rhs(u+dt*k1/2,r,n_qp_1D,mass,inv_mass,phi_val,phi_grad,phi_val_bd,hx,hy,wts,wts2d,d1,d2,fact_int,fact_bd,complem_fact,radius,pts2d_phi,pts2d_phi_bd,coriolis_fun,eq_type);
        k3=compute_rhs(u+dt*(1/2*k2),r,n_qp_1D,mass,inv_mass,phi_val,phi_grad,phi_val_bd,hx,hy,wts,wts2d,d1,d2,fact_int,fact_bd,complem_fact,radius,pts2d_phi,pts2d_phi_bd,coriolis_fun,eq_type);
        k4=compute_rhs(u+dt*(1*k3),r,n_qp_1D,mass,inv_mass,phi_val,phi_grad,phi_val_bd,hx,hy,wts,wts2d,d1,d2,fact_int,fact_bd,complem_fact,radius,pts2d_phi,pts2d_phi_bd,coriolis_fun,eq_type);
        u=u+dt*1/6*k1+dt*1/3*k2+dt*1/3*k3+dt*1/6*k4;   
    
    elseif RK>=5
        error('RK scheme not implemented');
    end
    
    if eq_type == "swe_sphere"
        
        
    end
    
    if debug
        if (mod(iter, debug_freq) == 0) && (eq_type == "swe" || eq_type == "swe_sphere")
            t = iter*dt;
            uN = modal2nodal(u(:,:,:),V_rect,r);
            fprintf('Iteration %d/%d Time: %.1f sec (%.1f h; %.1f days)\n',iter,N_it, t, t/3600, t/86400); 
            fprintf('Mean h = %.2f; Mean u = %.2f; Mean v = %.2f\n',mean(uN(:,:,1), 'all'), mean(uN(:,:,2)./uN(:,:,1), 'all'), mean(uN(:,:,3)./uN(:,:,1), 'all'));
            fprintf('Max h = %.2f; Max u = %.2f; Max v = %.2f\n\n',max(uN(:,:,1), [], 'all'), max(uN(:,:,2)./uN(:,:,1), [], 'all'), max(uN(:,:,3)./uN(:,:,1), [], 'all'));
            
        end
        
    end
    
    %plot solution
    if (iter - plot_freq >= 0 && mod(iter-1,plot_freq)==0) || iter==N_it
        t = iter*dt;
        uN = modal2nodal(u(:,:,:),V_rect,r);
        if eq_type == "swe" || eq_type == "swe_sphere"
            fprintf('Max v = %f  ', max(uN(:,:,3)./uN(:,:,1), [], 'all'));
        end
        fprintf('Iteration %d/%d Time: %.1f sec (%.1f h; %.1f days)\n',iter,N_it, t, t/3600, t/86400); 
        figure(f2);
        if eq_type == "swe" || eq_type == "swe_sphere"
            uN(:,:,2) = uN(:,:,2) ./ uN(:,:,1);
            uN(:,:,3) = uN(:,:,3) ./ uN(:,:,1);
        end

        plot_solution(uN,x_u(:),y_u(:),n_qp_1D-1,d1,d2,radius, ...
            plot_type, sprintf("Simulation: T = %.1f sec (%.2f h; %.1f days)", t, t/3600, t/86400), ...
            eq_type, fact_int);
        pause(0.05);
    end
    
    %next iteration
    if t+dt>T
        dt=T-t; 
    end
    t=t+dt;
    
    %avoid useless iterations if solution is already diverging
    if max(abs(u(:)))>=1e8
        fprintf('Iteration %d/%d  Time: %f\n',iter,N_it, iter*dt); error('Solution is diverging'); 
    end
    
end
toc

%plot all components of the solution
% for i=1:size(u,3)
%     figure(200); 
%     plot_solution(modal2nodal(u(:,:,:),V_rect,r),x_u,y_u,n_qp_1D-1,d1,d2,plot_type, eq_type, fact_int); 
% end




%compute discretization error, assuming that solution did one complete rotation
% if eq_type=="linear" || eq_type=="adv_sphere" || eq_type=="swe_sphere"
%     f3 = figure('units','normalized','outerposition',[0 0 1 1]);  
%     figure(f3);    
%     uN_0 = modal2nodal(nodal2modal(u0(:,:,:),V,r),V_rect,r);
%     uN_final = modal2nodal(u(:,:,:),V_rect,r);
%     
%     error_plot = zeros(n_qp, d1*d2, 3);
%     %%% ---ONLY First Component ---
% %     error_plot(:,:,1) = uN_0(:,:,1);
% %     error_plot(:,:,2) = uN_final(:,:,1);
% %     error_plot(:,:,3) = uN_final(:,:,1) - uN_0(:,:,1);
% %%% ----
% 
% %%% --- ALL Components ---
%     error_plot(:,:,1) = uN_final(:,:,1) - uN_0(:,:,1);
%     if eq_type == "swe_sphere"
%         u0_ = uN_0(:,:,2)./uN_0(:,:,1);
%         v0 = uN_0(:,:,3)./uN_0(:,:,1);
%         
%         uF = uN_final(:,:,2)./uN_final(:,:,1);
%         vF = uN_final(:,:,3)./uN_final(:,:,1);
%         
%         error_plot(:,:,2) = (uF - u0_);
%         error_plot(:,:,3) = (vF - v0);
% 
%     else
%         error_plot(:,:,2) = uN_final(:,:,2) - uN_0(:,:,2);
%         error_plot(:,:,3) = uN_final(:,:,3) - uN_0(:,:,3);
%     end
% 
%     
%     plot_solution(error_plot,x_u(:),y_u(:),n_qp_1D-1,d1,d2,radius,plot_type,"Final timestep error", eq_type, fact_int);

% === OUTPUT ===
%     errL2=reshape(u(:,:,1)-nodal2modal(u0(:,:,1),V,r),dim*d1*d2,1)'*mass*reshape(u(:,:,1)-nodal2modal(u0(:,:,1),V,r),dim*d1*d2,1);
%     normL2_sol=reshape(nodal2modal(u0(:,:,1),V,r),dim*d1*d2,1)'*mass*reshape(nodal2modal(u0(:,:,1),V,r),dim*d1*d2,1);
%     normL2_final=reshape(u,dim*d1*d2,1)'*mass*reshape(u,dim*d1*d2,1);
%     fprintf('L2 error is %f, and after normalization is %f with normalization %f\n',sqrt(errL2),sqrt(errL2/normL2_sol), sqrt(normL2_sol));
%     fprintf('Initial mass %f\n', sqrt(normL2_sol))
%     fprintf('Final mass %f\n', sqrt(normL2_final))
    
    
% end


fprintf('End of program\n');