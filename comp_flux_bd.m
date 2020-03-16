function [flux_n,flux_s,flux_e,flux_w] = compute_numerical_flux_new(u_n,u_s,u_e,u_w,pts_x,pts_y,d1,d2,neq,hx,hy,eq_type,radius,x_c,y_c)

flux_n = cell(d1,d2,neq); flux_s = cell(d1,d2,neq); flux_e = cell(d1,d2,neq); flux_w = cell(d1,d2,neq);

%uL is the solution in current element == u  (new allocation not needed)

% Find the U values from the N, S, E, W neighboring cells
%         

[fx_n, fy_n] = flux_function_new(u_n,eq_type,radius,hx,hy,x_c,y_c+hy/2,pts_x,zeros(size(pts_x)));
[fx_s, fy_s] = flux_function_new(u_s,eq_type,radius,hx,hy,x_c,y_c-hy/2,pts_x,zeros(size(pts_x)));
[fx_e, fy_e] = flux_function_new(u_e,eq_type,radius,hx,hy,x_c+hx/2,y_c,zeros(size(pts_y)),pts_y);
[fx_w, fy_w] = flux_function_new(u_w,eq_type,radius,hx,hy,x_c-hx/2,y_c,zeros(size(pts_y)),pts_y);

%convention: 1=bottom, 2=right, 3=top, 4=left
%alternatively:  N=3, S=1, E=2, W=4


%compute Lax-Friedrichs flux (multiplied with normal vector)
% Each case has a different maximum wave speeds (was: get_maximum_eig) over each face (same for all BD QP)

switch eq_type

    case "linear"

%
% TODO: confirm the signs for each face (5.3.20: looks correct to me)
        alpha = 1.0;
        for i=1:d1
            for j=1:d2
                in=i; jn=mod(j,d2)+1;     % Find the index of neighbor sharing north edge 
                is=i; js=mod(j-2,d2)+1;   % Find the index of neighbor sharing south edge 
                ie=mod(i,d1)+1;   je=j;   % Find the index of neighbor sharing east edge 
                iw=mod(i-2,d1)+1; jw=j;   % Find the index of neighbor sharing west edge 
                for n=1:neq
                    flux_n{i,j,n} =  1/2*(fy_n{i,j,n}+fy_s{in,jn,n}) - alpha/2 * (u_s{in,jn,n} - u_n{i,j,n});
                    flux_s{i,j,n} = -1/2*(fy_s{i,j,n}+fy_n{is,js,n}) - alpha/2 * (u_n{is,js,n} - u_s{i,j,n});
                    flux_e{i,j,n} =  1/2*(fx_e{i,j,n}+fx_w{ie,je,n}) - alpha/2 * (u_w{ie,je,n} - u_e{i,j,n});
                    flux_w{i,j,n} = -1/2*(fx_w{i,j,n}+fx_e{iw,jw,n}) - alpha/2 * (u_e{iw,jw,n} - u_w{i,j,n});
                end
            end
        end
%
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
                    flux_n{i,j,n} = 1/2*(fy_n{i,j,n}+fyR_n{i,j,n}) - alpha/2 * (uR_n{i,j,n} - u_n{i,j,n});
                end

                % Calculate the maximum wave speed over the south face (same for all QP)
                alphaL =  max( sqrt(abs(g*u_s{i,j,1}))+sqrt((u_s{i,j,2}./u_s{i,j,1}).^2+(u_s{i,j,3}./u_s{i,j,1}).^2) );
                alphaR =  max( sqrt(abs(g*uR_s{i,j,1}))+sqrt((uR_s{i,j,2}./uR_s{i,j,1}).^2+(uR_s{i,j,3}./uR_s{i,j,1}).^2) );
                alpha  =  max(alphaL, alphaR);
                for n=1:neq
                    flux_s{i,j,n} = -1/2*(fy_s{i,j,n}+fyR_s{i,j,n}) - alpha/2 * (uR_s{i,j,n} - u_s{i,j,n});
                end

                % Calculate the maximum wave speed over the east face (same for all QP)
                alphaL =  max( sqrt(abs(g*u_e{i,j,1}))+sqrt((u_e{i,j,2}./u_e{i,j,1}).^2+(u_e{i,j,3}./u_e{i,j,1}).^2) );
                alphaR =  max( sqrt(abs(g*uR_e{i,j,1}))+sqrt((uR_e{i,j,2}./uR_e{i,j,1}).^2+(uR_e{i,j,3}./uR_e{i,j,1}).^2) );
                alpha  =  max(alphaL, alphaR);
                for n=1:neq
                    flux_e{i,j,n} = 1/2*(fx_e{i,j,n}+fxR_e{i,j,n}) - alpha/2 * (uR_e{i,j,n} - u_e{i,j,n});
                end


                % Calculate the maximum wave speed over the west face (same for all QP)
                alphaL =  max( sqrt(abs(g*u_w{i,j,1}))+sqrt((u_w{i,j,2}./u_w{i,j,1}).^2+(u_w{i,j,3}./u_w{i,j,1}).^2) );
                alphaR =  max( sqrt(abs(g*uR_w{i,j,1}))+sqrt((uR_w{i,j,2}./uR_w{i,j,1}).^2+(uR_w{i,j,3}./uR_w{i,j,1}).^2) );
                alpha  =  max(alphaL, alphaR);
                for n=1:neq
                    flux_w{i,j,n} = -1/2*(fx_w{i,j,n}+fxR_w{i,j,n}) - alpha/2 * (uR_w{i,j,n} - u_w{i,j,n});
                end
            end
        end

    case "adv_sphere"
        angle = pi/2;
        cos_angle = 0.0;
	sin_angle = 1.0;
        for i=1:d1
            for j=1:d2
                qp_x=x_c(i)+pts_x*hx/2;
                qp_y=y_c(j)+pts_y*hy/2;
                qp_x_e = x_c(i)+hx/2; qp_x_w = x_c(i)-hx/2; % Scalars
                qp_y_n = y_c(j)+hy/2; qp_y_s = y_c(j)-hy/2; % Scalars
                beta_x=2*pi*radius/(12*86400)*(cos(qp_y)*cos_angle+sin(qp_y).*cos(qp_x)*sin_angle);      % Vector
                beta_x_n=2*pi*radius/(12*86400)*(cos(qp_y_n)*cos_angle+sin(qp_y_n)*cos(qp_x)*sin_angle); % Vector
                beta_x_s=2*pi*radius/(12*86400)*(cos(qp_y_s)*cos_angle+sin(qp_y_s)*cos(qp_x)*sin_angle); % Vector
                beta_x_e=2*pi*radius/(12*86400)*(cos(qp_y)*cos_angle+sin(qp_y)*cos(qp_x_e)*sin_angle);   % Vector
                beta_x_w=2*pi*radius/(12*86400)*(cos(qp_y)*cos_angle+sin(qp_y)*cos(qp_x_w)*sin_angle);   % Vector
                beta_y=-2*pi*radius/(12*86400)*sin_angle*sin(qp_x);     % Vector
                beta_y_e=-2*pi*radius/(12*86400)*sin_angle*sin(qp_x_e); % Scalar
                beta_y_w=-2*pi*radius/(12*86400)*sin_angle*sin(qp_x_w); % Scalar
                alpha_n = max( sqrt(beta_x_n.^2+beta_y.^2), [], 1); alpha_s = max( sqrt(beta_x_s.^2+beta_y.^2), [], 1);
                alpha_e = max( sqrt(beta_x_e.^2+beta_y_e^2), [], 1); alpha_w = max( sqrt(beta_x_w.^2+beta_y_w^2), [], 1);

                if ( j==d2 )
                    fact_bd_n = 0.0;           % Avoid epsilon values in north pol
                else
                    fact_bd_n = cos(qp_y_n);  % Scalar
                end
                if ( j==1 )
                    fact_bd_s = 0.0;
                else
		    fact_bd_s = cos(qp_y_s);  % Scalar
                end
% TODO: confirm the signs for each face (5.3.20: looks correct to me)
% TODO: check the fact_bd factor...  Why is this not simply a scalar???  Why does it not apply to EW fluxes??
                in=i; jn=mod(j,d2)+1;     % Find the index of neighbor sharing north edge 
                is=i; js=mod(j-2,d2)+1;   % Find the index of neighbor sharing south edge 
                ie=mod(i,d1)+1;   je=j;   % Find the index of neighbor sharing east edge 
                iw=mod(i-2,d1)+1; jw=j;   % Find the index of neighbor sharing west edge 
                for n=1:neq
                    flux_n{i,j,n} =  1/2*(fy_n{i,j,n}+fy_s{in,jn,n})*fact_bd_n - alpha_n/2 *(u_s{in,jn,n} - u_n{i,j,n})*fact_bd_n; 
                    flux_s{i,j,n} = -1/2*(fy_s{i,j,n}+fy_n{is,js,n})*fact_bd_s - alpha_s/2 *(u_n{is,js,n} - u_s{i,j,n})*fact_bd_s;
                    flux_e{i,j,n} =  1/2*(fx_e{i,j,n}+fx_w{ie,je,n}) - alpha_e/2 * (u_w{ie,je,n} - u_e{i,j,n});
                    flux_w{i,j,n} = -1/2*(fx_w{i,j,n}+fx_e{iw,jw,n}) - alpha_w/2 * (u_e{iw,jw,n} - u_w{i,j,n});
                end
            end
	end

    case "swe_sphere"

% Not yet implemented.

end

end
