function [flux_n,flux_s,flux_e,flux_w] = compute_numerical_flux_new(u_n,u_s,u_e,u_w,pts2d_x,pts2d_y,d1,d2,neq,hx,hy,eq_type,radius,x_c,y_c)

flux_n = cell(d1,d2,neq);
flux_s = cell(d1,d2,neq);
flux_e = cell(d1,d2,neq);
flux_w = cell(d1,d2,neq);

%uL is the solution in current element == u  (new allocation not needed)
%uR is the solution in the neighboring element
uR_n = cell(d1,d2,neq);
uR_s = cell(d1,d2,neq);
uR_e = cell(d1,d2,neq);
uR_w = cell(d1,d2,neq);

%
% Find the U values from the N, S, E, W neighboring cells
%         

for i=1:d1
    for j=1:d2
        for n=1:neq
            uR_n{i,j,n} = u_s{i,j+1,n};
            uR_s{i,j,n} = u_n{i,j-1,n};
            uR_e{i,j,n} = u_w{i-1,j,n};
            uR_w{i,j,n} = u_e{i+1,j,n}
        end
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

%convention: 1=bottom, 2=right, 3=top, 4=left
%alternatively:  N=3, S=1, E=2, W=4

%%% normal_x(:,:,1)=0; normal_x(:,:,2)=1; normal_x(:,:,3)=0; normal_x(:,:,4)=-1;
%%% normal_y(:,:,1)=-1; normal_y(:,:,2)=0; normal_y(:,:,3)=1; normal_y(:,:,4)=0;


% TODO delete this
%%%flux=nan(size(u));
%%%for n=1:size(u,4)
%%%    flux(:,:,:,n)=1/2*((fxL(:,:,:,n)+fxR(:,:,:,n)).*normal_x+(fyL(:,:,:,n)+fyR(:,:,:,n)).*fact_bd.*normal_y)...
%%%        -alpha/2.*(uR(:,:,:,n)-uL(:,:,:,n)).*(normal_x.^2+fact_bd.*normal_y.^2);
%%%end
%compute Lax-Friedrichs flux (multiplied with normal vector)
% Each case has a different maximum wave speeds (was: get_maximum_eig) over each face (same for all BD QP)

switch eq_type

    case "linear"

%
% TODO: confirm the signs for each face (5.3.20: looks correct to me)
        alpha = 1.0;
        for i=1:d1
            for j=1:d2
                for n=1:neq
                    flux_n{i,j,n} =  1/2*(fxL_n{i,j,n}+fxR_n{i,j,n}) - alpha/2 * (uR_n{i,j,n} - u_n{i,j,n});
                    flux_s{i,j,n} = -1/2*(fxL_s{i,j,n}+fxR_s{i,j,n}) - alpha/2 * (uR_s{i,j,n} - u_s{i,j,n});
                    flux_e{i,j,n} =  1/2*(fxL_e{i,j,n}+fxR_e{i,j,n}) - alpha/2 * (uR_e{i,j,n} - u_e{i,j,n});
                    flux_w{i,j,n} = -1/2*(fxL_w{i,j,n}+fxR_w{i,j,n}) - alpha/2 * (uR_w{i,j,n} - u_w{i,j,n});
                end
            end
        end

    case "swe"

        g=9.80616;
        for i=1:d1
            for j=1:d2

%%% TODO:  create a Lambda to do calculate alpha?

                % Calculate the maximum wave speed over the north face (same for all QP)
                alphaL =  max( sqrt(abs(g*u_n{i,j,1}))+sqrt((u_n{i,j,2}./u_n{i,j,1}).^2+(u_n{i,j,3}./u_n{i,j,1}).^2) );
                alphaR =  max( sqrt(abs(g*uR_n{i,j,1}))+sqrt((uR_n{i,j,2}./uR_n{i,j,1}).^2+(uR_n{i,j,3}./uR_n{i,j,1}).^2) );
                alpha  =  max(alphaL, alphaR);
                for n=1:neq
                    flux_n{i,j,n} = 1/2*(fxL_n{i,j,n}+fxR_n{i,j,n}) - alpha/2 * (uR_n{i,j,n} - u_n{i,j,n});
                end

                % Calculate the maximum wave speed over the south face (same for all QP)
                alphaL =  max( sqrt(abs(g*u_s{i,j,1}))+sqrt((u_s{i,j,2}./u_s{i,j,1}).^2+(u_s{i,j,3}./u_s{i,j,1}).^2) );
                alphaR =  max( sqrt(abs(g*uR_s{i,j,1}))+sqrt((uR_s{i,j,2}./uR_s{i,j,1}).^2+(uR_s{i,j,3}./uR_s{i,j,1}).^2) );
                alpha  =  max(alphaL, alphaR);
                for n=1:neq
                    flux_s{i,j,n} = 1/2*(fxL_s{i,j,n}+fxR_s{i,j,n}) - alpha/2 * (uR_s{i,j,n} - u_s{i,j,n});
                end

                % Calculate the maximum wave speed over the east face (same for all QP)
                alphaL =  max( sqrt(abs(g*u_e{i,j,1}))+sqrt((u_e{i,j,2}./u_e{i,j,1}).^2+(u_e{i,j,3}./u_e{i,j,1}).^2) );
                alphaR =  max( sqrt(abs(g*uR_e{i,j,1}))+sqrt((uR_e{i,j,2}./uR_e{i,j,1}).^2+(uR_e{i,j,3}./uR_e{i,j,1}).^2) );
                alpha  =  max(alphaL, alphaR);
                for n=1:neq
                    flux_e{i,j,n} = 1/2*(fxL_e{i,j,n}+fxR_e{i,j,n}) - alpha/2 * (uR_e{i,j,n} - u_e{i,j,n});
                end


                % Calculate the maximum wave speed over the west face (same for all QP)
                alphaL =  max( sqrt(abs(g*u_w{i,j,1}))+sqrt((u_w{i,j,2}./u_w{i,j,1}).^2+(u_w{i,j,3}./u_w{i,j,1}).^2) );
                alphaR =  max( sqrt(abs(g*uR_w{i,j,1}))+sqrt((uR_w{i,j,2}./uR_w{i,j,1}).^2+(uR_w{i,j,3}./uR_w{i,j,1}).^2) );
                alpha  =  max(alphaL, alphaR);
                for n=1:neq
                    flux_w{i,j,n} = 1/2*(fxL_w{i,j,n}+fxR_w{i,j,n}) - alpha/2 * (uR_w{i,j,n} - u_w{i,j,n});
                end
            end
        end

    case "adv_sphere"

        for i=1:d1
            for j=1:d2
                qp_y=y_c(j)+pts2d_y/2*hy;
                % Calculate fact_bd on the fly
		fact_bd = cos(qp_y);
% TODO: confirm the signs for each face (5.3.20: looks correct to me)
                for n=1:neq
                    flux_n{i,j,n} =  1/2*(fxL_n{i,j,n}+fxR_n{i,j,n}).*fact_bd - alpha/2 * (uR_n{i,j,n} - u_n{i,j,n}) .* fact_bd; 
                    flux_s{i,j,n} = -1/2*(fxL_s{i,j,n}+fxR_s{i,j,n}).*fact_bd - alpha/2 * (uR_s{i,j,n} - u_s{i,j,n}) .* fact_bd;
                    flux_e{i,j,n} =  1/2*(fxL_e{i,j,n}+fxR_e{i,j,n}) - alpha/2 * (uR_e{i,j,n} - u_e{i,j,n});
                    flux_w{i,j,n} = -1/2*(fxL_w{i,j,n}+fxR_w{i,j,n}) - alpha/2 * (uR_w{i,j,n} - u_w{i,j,n});
                end
            end
	end

    case "swe_sphere"

% Not yet implemented.

end

end
