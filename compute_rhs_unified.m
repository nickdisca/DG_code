function [rhs_u] = compute_rhs_unified(u,r,mass,inv_mass,phi_val,phi_grad,phi_val_bd,...
                                       fact_int,fact_bd,complem_fact,pts2d_phi,pts2d_phi_bd,...
                                       u_new,r_new,n_qp_1D,phi_val_cell,phi_grad_cell_x,phi_grad_cell_y,...
                                       phi_val_bd_cell_n,phi_val_bd_cell_s,phi_val_bd_cell_e,phi_val_bd_cell_w,inv_mass_new,...
                                       hx,hy,wts,wts2d,radius,pts_x,pts_y,pts2d_x,pts2d_y,x_c,y_c,coriolis_fun,eq_type)
%compute rsh of the ode

d1=size(u_new,1);
d2=size(u_new,2);
neq=size(u_new,3);

for i=1:d1
    for j=1:d2
        abs_err = norm(u(:,(i-1)*d2+j,1) - u_new{i,j,1});
        if abs_err > 1e-9
            u_new_error = [ i j abs_err ]
        end
    end
end

%dimension: (cardinality)*(num_elems)*(num_eqns)
rhsu=zeros(size(u));

%cardinality and qp
dim=(max(r(:))+1)^2;
n_qp=n_qp_1D^2;

%determinants of the internal and boundary mappings: 1=bottom, 2=right, 3=top, 4=left
determ=hx*hy/4;
bd_det(1)=hx/2; bd_det(2)=hy/2; bd_det(3)=hx/2; bd_det(4)=hy/2;

%compute solution in the quadrature points (sort of modal to nodal
%conversion)
u_qp=zeros(n_qp,size(u,2),size(u,3));
for i=1:size(u,3)
    u_qp(:,:,i)=reshape(phi_val{1}*reshape(u(:,:,i),dim*d1*d2,1),n_qp,d1*d2);
    %u_qp(:,:,i)=phi_val*u(:,:,i);
end

%INTERNAL INTEGRALS

%compute physical fluxes
[flux_fun_x, flux_fun_y]=flux_function(u_qp,eq_type,radius,pts2d_phi(1:n_qp,:),pts2d_phi(n_qp+1:2*n_qp,:));

%compute internal integral and add it to the rhs
%det_internal*inverse_of_jacobian_x_affine*sum(dPhi/dr*f_x*weights)+
%det_internal*inverse_of_jacobian_y_affine*sum(dPhi/ds*f_y*weights)
for i=1:size(u,3)
    rhsu(:,:,i)=reshape(phi_grad{1}'*reshape(flux_fun_x(:,:,i).*wts2d,n_qp*d1*d2,1),dim,d1*d2)*(2/hx)*determ +...
        reshape(phi_grad{2}'*reshape(fact_int.*flux_fun_y(:,:,i).*wts2d,n_qp*d1*d2,1),dim,d1*d2)*(2/hy)*determ;
    %rhsu(:,:,i)=phi_grad(:,:,1)'*(flux_fun_x(:,:,i).*wts2d)*(2/hx)*determ +...
        %phi_grad(:,:,2)'*(fact_int.*flux_fun_y(:,:,i).*wts2d)*(2/hy)*determ;
end

%BOUNDARY INTEGRALS

%compute solution in the boundary quadrature points (sort of modal to nodal
%conversion)
%dimension: (num_bd_qp)*(cardinality)*num_faces*num_equations
u_qp_bd=zeros(length(wts),size(u,2),4,size(u,3));
for n=1:size(u,3)
    for i=1:4
        u_qp_bd(:,:,i,n)=reshape(phi_val_bd{i}*reshape(u(:,:,n),dim*d1*d2,1),n_qp_1D,d1*d2);
        %u_qp_bd(:,:,i,n)=phi_val_bd(:,:,i)*u(:,:,n); 
    end
end

%compute LF fluxes
num_flux=compute_numerical_flux(u_qp_bd,d1,d2,fact_bd,eq_type,radius,pts2d_phi_bd(1:n_qp_1D,:,:),pts2d_phi_bd(n_qp_1D+1:2*n_qp_1D,:,:));

%compute boundary integrals and subtract them to the rhs
bd_term=zeros(size(u));
for n=1:size(u,3)
    for i=1:4
        %det_bd*sum(Phi*num_flux*weights)
        bd_term(:,:,n)=bd_term(:,:,n)+reshape(phi_val_bd{i}'*reshape(num_flux(:,:,i,n).*wts,n_qp_1D*d1*d2,1),dim,d1*d2)*bd_det(i);
        %bd_term(:,:,n)=bd_term(:,:,n)+phi_val_bd(:,:,i)'*(num_flux(:,:,i,n).*wts)*bd_det(i);
    end
    rhsu(:,:,n)=rhsu(:,:,n)-bd_term(:,:,n);
end

%invert the (local) mass matrix and divide by radius
for n=1:size(u,3)
    rhsu(:,:,n)=1/radius*reshape(inv_mass*reshape(rhsu(:,:,n),dim*d1*d2,1),dim,d1*d2);
end



%
% NEW section
%

rsh_u = cell(size(u_new));

%
% Allocate RHS
for i=1:d1
    for j=1:d2
        for n=1:neq
            rhs_u{i,j,n}=zeros(size(u_new{i,j,n}));
        end
    end
end

%cardinality and qp
dim=(max(r_new(:))+1)^2;
n_qp=n_qp_1D^2;

%determinants of the internal and boundary mappings: 1=bottom, 2=right, 3=top, 4=left
determ=hx*hy/4;
bd_det(1)=hx/2; bd_det(2)=hy/2; bd_det(3)=hx/2; bd_det(4)=hy/2;

%compute solution in the quadrature points (sort of modal to nodal
%conversion)
u_qp_new=cell(size(u_new));


for i=1:d1
    for j=1:d2
        for n=1:neq
            u_qp_new{i,j,n}=phi_val_cell{r_new(i,j)}*u_new{i,j,n};
        end
    end
end

%INTERNAL INTEGRALS

%compute physical value of F(x) inside the region 
[flux_fun_x_new, flux_fun_y_new]=flux_function_new(u_qp_new,eq_type,radius,hx,hy,x_c,y_c,pts2d_x,pts2d_y);

for i=1:d1
    for j=1:d2
        abs_err = norm(flux_fun_x(:,(i-1)*d2+j,1) - flux_fun_x_new{i,j,1});
        if abs_err > 1e-9
            flux_fun_x_error = [ i j abs_err ]
        end
        abs_err = norm(flux_fun_y(:,(i-1)*d2+j,1) - flux_fun_y_new{i,j,1});
        if abs_err > 1e-9
            flux_fun_y_error = [ i j abs_err ]
        end
    end
end


%compute internal integral and add it to the rhs
%det_internal*inverse_of_jacobian_x_affine*sum(dPhi/dr*f_x*weights)+
%det_internal*inverse_of_jacobian_y_affine*sum(dPhi/ds*f_y*weights)

if eq_type=="linear" || eq_type=="swe"
% Cartesian geometry
    for i=1:d1
        for j=1:d2
            for n=1:neq
                rhs_u{i,j,n}=phi_grad_cell_x{r_new(i,j)}'*(flux_fun_x_new{i,j,n}.*wts2d)*(2/hx)*determ +...
                            phi_grad_cell_y{r_new(i,j)}'*(flux_fun_y_new{i,j,n}.*wts2d)*(2/hy)*determ;
            end
        end
    end
end

if eq_type=="adv_sphere" || eq_type=="swe_sphere"
%Spherical geometry
    for i=1:d1
        for j=1:d2
            for n=1:neq
%calculate qp_y on the fly
                qp_y=y_c(j)+pts2d_y/2*hy;
                rhs_u{i,j,n}=phi_grad_cell_x{r_new(i,j)}'*(flux_fun_x_new{i,j,n}.*wts2d)*(2/hx)*determ +...
                            phi_grad_cell_y{r_new(i,j)}'*(cos(qp_y).*flux_fun_y_new{i,j,n}.*wts2d)*(2/hy)*determ;
%                max_rhs_u=[i j max(rhsu{i,j,n})]
            end
        end
    end
end


%BOUNDARY INTEGRALS

%compute solution in the boundary quadrature points (sort of modal to nodal
%conversion)
%dimension: (num_bd_qp)*(cardinality)*num_faces*num_equations

u_qp_bd_n=cell(d1,d2,neq);
u_qp_bd_s=cell(d1,d2,neq);
u_qp_bd_e=cell(d1,d2,neq);
u_qp_bd_w=cell(d1,d2,neq);

for i=1:d1
    for j=1:d2
        for n=1:neq
            u_qp_bd_n{i,j,n} = phi_val_bd_cell_n{r_new(i,j)}*u_new{i,j,n}; 
            u_qp_bd_s{i,j,n} = phi_val_bd_cell_s{r_new(i,j)}*u_new{i,j,n}; 
            u_qp_bd_e{i,j,n} = phi_val_bd_cell_e{r_new(i,j)}*u_new{i,j,n}; 
            u_qp_bd_w{i,j,n} = phi_val_bd_cell_w{r_new(i,j)}*u_new{i,j,n}; 
        end
    end
end

for i=1:d1
    for j=1:d2
        abs_err = norm(u_qp_bd(:,(i-1)*d2+j,3,1) - u_qp_bd_n{i,j,1});
        if abs_err > 1e-9
            u_qp_bd_n_error = [ i j abs_err ]
        end
        abs_err = norm(u_qp_bd(:,(i-1)*d2+j,1,1) - u_qp_bd_s{i,j,1});
        if abs_err > 1e-9
            u_qp_bd_s_error = [ i j abs_err ]
        end
        abs_err = norm(u_qp_bd(:,(i-1)*d2+j,2,1) - u_qp_bd_e{i,j,1});
        if abs_err > 1e-9
            u_qp_bd_e_error = [ i j abs_err ]
        end
        abs_err = norm(u_qp_bd(:,(i-1)*d2+j,4,1) - u_qp_bd_w{i,j,1});
        if abs_err > 1e-9
            u_qp_bd_w_error = [ i j abs_err ]
        end
    end
end

%compute LF fluxes on all four edges
%  First cut: calculate all edges simultaneously
[flux_n,flux_s,flux_e,flux_w]=comp_flux_bd(u_qp_bd_n,u_qp_bd_s,u_qp_bd_e,u_qp_bd_w,...
                                           pts_x,pts_y,d1,d2,neq,hx,hy,eq_type,radius,x_c,y_c);

for i=1:d1
    for j=1:d2
        junk = num_flux(:,(i-1)*d2+j,3,1) - flux_n{i,j,1};
        abs_err = norm(junk);
        if abs_err > 1e-9
            flux_n_error = [ i j num_flux(:,(i-1)*d2+j,3,1)' flux_n{i,j,1}' ]
        end
        junk = num_flux(:,(i-1)*d2+j,1,1) - flux_s{i,j,1};
        abs_err = norm(junk);
        if abs_err > 1e-9
            flux_s_error = [ i j num_flux(:,(i-1)*d2+j,1,1)' flux_s{i,j,1}' ]
        end
        junk = num_flux(:,(i-1)*d2+j,2,1) - flux_e{i,j,1};
        abs_err = norm(junk);
        if abs_err > 1e-9
            flux_e_error = [ i j num_flux(:,(i-1)*d2+j,2,1)' flux_e{i,j,1}' ]
        end
        junk = num_flux(:,(i-1)*d2+j,4,1) - flux_w{i,j,1};
        abs_err = norm(junk);
        if abs_err > 1e-9
            flux_w_error = [ i j num_flux(:,(i-1)*d2+j,4,1)' flux_w{i,j,1}']
        end
    end
end

%dimension: (cardinality)*(num_elems)*(num_eqns)

for i=1:d1
    for j=1:d2
        for n=1:neq
            rhs_u{i,j,n} = rhs_u{i,j,n} - 0.5*hx*phi_val_bd_cell_n{r_new(i,j)}'*(flux_n{i,j,n}.*wts)... 
                                        - 0.5*hx*phi_val_bd_cell_s{r_new(i,j)}'*(flux_s{i,j,n}.*wts)...
                                        - 0.5*hy*phi_val_bd_cell_e{r_new(i,j)}'*(flux_e{i,j,n}.*wts)...
                                        - 0.5*hy*phi_val_bd_cell_w{r_new(i,j)}'*(flux_w{i,j,n}.*wts);
        end
    end
end


%add corrective internal term for the divergence, only in SWE spherical coordinates
if eq_type=="swe_sphere"

						 % NOT CURRENTLY SUPPORTED
						 
    %rhsu(:,:,2)=rhsu(:,:,2)+phi_val'*(complem_fact.*flux_fun_y(:,:,2).*wts2d)*determ;
    %rhsu(:,:,3)=rhsu(:,:,3)-phi_val'*(complem_fact.*flux_fun_x(:,:,2).*wts2d)*determ;
end


%add coriolis term to second and third equation of the swe
if eq_type=="swe" || eq_type=="swe_sphere"

						 % NOT CURRENTLY SUPPORTED

%%%    coriolis=coriolis_fun(pts2d_phi(1:n_qp,:),pts2d_phi(n_qp+1:2*n_qp,:));
    %rhsu(:,:,2)=rhsu(:,:,2)+radius*phi_val(:,:)'*(fact_int.*coriolis.*u_qp(:,:,3).*wts2d)*determ;
	%rhsu(:,:,3)=rhsu(:,:,3)-radius*phi_val(:,:)'*(fact_int.*coriolis.*u_qp(:,:,2).*wts2d)*determ;
end

%invert the (local) mass matrix and divide by radius

for i=1:d1
    for j=1:d2
        for n=1:neq
            rhs_u{i,j,n} = 1/radius * inv_mass_new{i,j}*rhs_u{i,j,n};
        end
    end
end

end
