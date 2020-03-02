function [flux] = compute_numerical_flux_new(u_n,u_s,u_e,u_w,pts2d_x,pts2d_y,eq_type,radius,pts,x_c,y_c)

%uL is the solution in current element
%uR is the solution in the neighboring element
uR_n = cell(size(u_n));
uR_s = cell(size(u_s));
uR_e = cell(size(u_e));
uR_w = cell(size(u_w));

%
% TODO:  implementation for N, S, E, W which are now explicit.
%         

for i=1:d1
    for j=1:d2
        uR_n{i,j} = u_s{i,j+1};
        uR_s{i,j) = u_n{i,j-1};
        uR_e{i,j} = u_w{i-1,j};
        uR_w{i,j} = u_e{i+1,j}
    end
end

[fxL_n, fyL_n] = flux_function_new(u_n,eq_type,radius,hx,hy,x_c,y_c+hy/2,pts2d_x,pts2d_y);
[fxL_s, fyL_s] = flux_function_new(u_s,eq_type,radius,hx,hy,x_c,y_c-hy/2,pts2d_x,pts2d_y);
[fxL_e, fyL_e] = flux_function_new(u_e,eq_type,radius,hx,hy,x_c+hx/2,y_c,pts2d_x,pts2d_y);
[fxL_w, fyL_w] = flux_function_new(u_w,eq_type,radius,hx,hy,x_c-hx/2,y_c,pts2d_x,pts2d_y);
[fxR_n, fyR_n] = flux_function_new(uR_n,eq_type,radius,hx,hy,x_c,y_c+hy/2,pts2d_x,pts2d_y);
[fxR_s, fyR_s] = flux_function_new(uR_s,eq_type,radius,hx,hy,x_c,y_c-hy/2,pts2d_x,pts2d_y);
[fxR_e, fyR_e] = flux_function_new(uR_e,eq_type,radius,hx,hy,x_c+hx/2,y_c,pts2d_x,pts2d_y);
[fxR_w, fyR_w] = flux_function_new(uR_w,eq_type,radius,hx,hy,x_c-hx/2,y_c,pts2d_x,pts2d_y);

% TODO: these can be deleted eventually since the fluxes are separated by face
%compute normal vectors for all faces: 1=bottom, 2=right, 3=top, 4=left
normal_x=zeros(size(u,1),size(u,2),4); 
normal_y=zeros(size(u,1),size(u,2),4);
normal_x(:,:,1)=0; normal_x(:,:,2)=1; normal_x(:,:,3)=0; normal_x(:,:,4)=-1;
normal_y(:,:,1)=-1; normal_y(:,:,2)=0; normal_y(:,:,3)=1; normal_y(:,:,4)=0;

% TODO get_maximum_eig_new
%compute maximum wave speed for all faces
alpha = max( get_maximum_eig(uL,eq_type,radius,qp_x,qp_y), get_maximum_eig(uR,eq_type,radius,qp_x,qp_y));

% TODO: delete this
flux=nan(size(u));
for n=1:size(u,4)
    flux(:,:,:,n)=1/2*((fxL(:,:,:,n)+fxR(:,:,:,n)).*normal_x+(fyL(:,:,:,n)+fyR(:,:,:,n)).*fact_bd.*normal_y)...
        -alpha/2.*(uR(:,:,:,n)-uL(:,:,:,n)).*(normal_x.^2+fact_bd.*normal_y.^2);
end
%compute Lax-Friedrichs flux (multiplied with normal vector)
if eq_type=="linear" || eq_type=="swe"

% TODO: complete this
    for i=1:d1
        for j=1:d2
            flux_n{i,j} = 
            flux_s{i,j} = 
            flux_e{i,j} = 
            flux_w{i,j} = 
        end
    end
end

if eq_type=="adv_sphere" || eq_type=="swe_sphere"
% TODO: complete this
% TODO: treat fact_bd using coordinates
    for i=1:d1
        for j=1:d2
            flux_n{i,j} = 
            flux_s{i,j} = 
            flux_e{i,j} = 
            flux_w{i,j} = 
        end
    end

end


end
