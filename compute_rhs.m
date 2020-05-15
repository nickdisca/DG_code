function [rhsu] = compute_rhs(u,r,n_qp_1D,phi_val_cell,phi_grad_cell_x,phi_grad_cell_y,...
                              phi_val_bd_cell_n,phi_val_bd_cell_s,phi_val_bd_cell_e,phi_val_bd_cell_w,inv_mass,...
                              hx,hy,wts,wts2d,radius,pts_x,pts_y,pts2d_x,pts2d_y,x_c,y_c,coriolis_fun,eq_type)
%compute rsh of the ode

%dimension: (cardinality)*(num_elems)*(num_eqns)
d1=size(u,1);
d2=size(u,2);
neq=size(u,3);

rshu = cell(size(u));

%
% Allocate RHS
for i=1:d1
    for j=1:d2
        for n=1:neq
            rhsu{i,j,n}=zeros(size(u{i,j,n}));
        end
    end
end

%cardinality and qp
dim=(max(r(:))+1)^2;
n_qp=n_qp_1D^2;

%determinants of the internal and boundary mappings: 1=bottom, 2=right, 3=top, 4=left
determ=hx*hy/4;
bd_det(1)=hx/2; bd_det(2)=hy/2; bd_det(3)=hx/2; bd_det(4)=hy/2;

%compute solution in the quadrature points (sort of modal to nodal
%conversion)
u_qp=cell(size(u));

for i=1:d1
    for j=1:d2
        for n=1:neq
            u_qp{i,j,n}=phi_val_cell{r(i,j)}*u{i,j,n};
        end
    end
end

%INTERNAL INTEGRALS

%compute physical value of F(x) inside the region 
[flux_fun_x, flux_fun_y]=flux_function(u_qp,eq_type,radius,hx,hy,x_c,y_c,pts2d_x,pts2d_y);

%compute internal integral and add it to the rhs
%det_internal*inverse_of_jacobian_x_affine*sum(dPhi/dr*f_x*weights)+
%det_internal*inverse_of_jacobian_y_affine*sum(dPhi/ds*f_y*weights)

if eq_type=="linear" || eq_type=="swe"
% Cartesian geometry
    for i=1:d1
        for j=1:d2
            for n=1:neq
                rhsu{i,j,n}=phi_grad_cell_x{r(i,j)}'*(flux_fun_x{i,j,n}.*wts2d)*(2/hx)*determ +...
                            phi_grad_cell_y{r(i,j)}'*(flux_fun_y{i,j,n}.*wts2d)*(2/hy)*determ;
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
                rhsu{i,j,n}=phi_grad_cell_x{r(i,j)}'*(flux_fun_x{i,j,n}.*wts2d)*(2/hx)*determ +...
                            phi_grad_cell_y{r(i,j)}'*(cos(qp_y).*flux_fun_y{i,j,n}.*wts2d)*(2/hy)*determ;
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
            u_qp_bd_n{i,j,n} = phi_val_bd_cell_n{r(i,j)}*u{i,j,n}; 
            u_qp_bd_s{i,j,n} = phi_val_bd_cell_s{r(i,j)}*u{i,j,n}; 
            u_qp_bd_e{i,j,n} = phi_val_bd_cell_e{r(i,j)}*u{i,j,n}; 
            u_qp_bd_w{i,j,n} = phi_val_bd_cell_w{r(i,j)}*u{i,j,n}; 
        end
    end
end

%compute LF fluxes on all four edges
%  First cut: calculate all edges simultaneously
[flux_n,flux_s,flux_e,flux_w]=comp_flux_bd(u_qp_bd_n,u_qp_bd_s,u_qp_bd_e,u_qp_bd_w,...
                                           pts_x,pts_y,d1,d2,neq,hx,hy,eq_type,radius,x_c,y_c);
for i=1:d1
    for j=1:d2
        for n=1:neq
            rhsu{i,j,n} = rhsu{i,j,n} - 0.5*hx*phi_val_bd_cell_n{r(i,j)}'*(flux_n{i,j,n}.*wts)... 
                                      - 0.5*hx*phi_val_bd_cell_s{r(i,j)}'*(flux_s{i,j,n}.*wts)...
                                      - 0.5*hy*phi_val_bd_cell_e{r(i,j)}'*(flux_e{i,j,n}.*wts)...
                                      - 0.5*hy*phi_val_bd_cell_w{r(i,j)}'*(flux_w{i,j,n}.*wts);
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
            rhsu{i,j,n} = 1/radius * inv_mass{i,j}*rhsu{i,j,n};
        end
    end
end

end
