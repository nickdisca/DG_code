function [flux_n,flux_s,flux_e,flux_w] = compute_numerical_flux_new(u_n,u_s,u_e,u_w,pts_x,pts_y,d1,d2,neq,hx,hy,eq_type,radius,x_c,y_c)

flux_n = cell(d1,d2,neq); flux_s = cell(d1,d2,neq); flux_e = cell(d1,d2,neq); flux_w = cell(d1,d2,neq);

%uL is the solution in current element == u  (new allocation not needed)
%uR is the solution in the neighboring element
										
uR_n = cell(d1,d2,neq); uR_s = cell(d1,d2,neq); uR_e = cell(d1,d2,neq); uR_w = cell(d1,d2,neq);
fxR_n = cell(d1,d2,neq); fxR_s = cell(d1,d2,neq); fxR_e = cell(d1,d2,neq); fxR_w = cell(d1,d2,neq);
fyR_n = cell(d1,d2,neq); fyR_s = cell(d1,d2,neq); fyR_e = cell(d1,d2,neq); fyR_w = cell(d1,d2,neq);

%
% Find the U values from the N, S, E, W neighboring cells
%         

[fx_n, fy_n] = flux_function_new(u_n,eq_type,radius,hx,hy,x_c,y_c+hy/2,pts_x,zeros(size(pts_x)));
[fx_s, fy_s] = flux_function_new(u_s,eq_type,radius,hx,hy,x_c,y_c-hy/2,pts_x,zeros(size(pts_x)));
[fx_e, fy_e] = flux_function_new(u_e,eq_type,radius,hx,hy,x_c+hx/2,y_c,zeros(size(pts_y)),pts_y);
[fx_w, fy_w] = flux_function_new(u_w,eq_type,radius,hx,hy,x_c-hx/2,y_c,zeros(size(pts_y)),pts_y);

%%%% deal with boundary cases separately.
for i=2:d1-1
    for j=2:d2-1
        for n=1:neq
            uR_n{i,j,n} = u_s{i,j+1,n}; uR_s{i,j,n} = u_n{i,j-1,n}; uR_e{i,j,n} = u_w{i+1,j,n}; uR_w{i,j,n} = u_e{i-1,j,n};
            fxR_n{i,j,n} = fx_s{i,j+1,n}; fxR_s{i,j,n} = fx_n{i,j-1,n}; fxR_e{i,j,n} = fx_w{i+1,j,n}; fxR_w{i,j,n} = fx_e{i-1,j,n};
            fyR_n{i,j,n} = fy_s{i,j+1,n}; fyR_s{i,j,n} = fy_n{i,j-1,n}; fyR_e{i,j,n} = fy_w{i+1,j,n}; fyR_w{i,j,n} = fy_e{i-1,j,n};
        end
    end
end

% Northmost and southmost cells
    for i=2:d1-1
        for n=1:neq
% Northmost
            uR_n{i,d2,n} = u_s{i,1,n}; uR_s{i,d2,n} = u_n{i,d2-1,n}; uR_e{i,d2,n} = u_w{i+1,d2,n}; uR_w{i,d2,n} = u_e{i-1,d2,n};
            fxR_n{i,d2,n} = fx_s{i,1,n}; fxR_s{i,d2,n} = fx_n{i,d2-1,n}; fxR_e{i,d2,n} = fx_w{i+1,d2,n}; fxR_w{i,d2,n} = fx_e{i-1,d2,n};
            fyR_n{i,d2,n} = fy_s{i,1,n}; fyR_s{i,d2,n} = fy_n{i,d2-1,n}; fyR_e{i,d2,n} = fy_w{i+1,d2,n}; fyR_w{i,d2,n} = fy_e{i-1,d2,n};

% Southmost
            uR_n{i,1,n} = u_s{i,2,n}; uR_s{i,1,n} = u_n{i,d2,n}; uR_e{i,1,n} = u_w{i+1,1,n}; uR_w{i,1,n} = u_e{i-1,1,n};
            fxR_n{i,1,n} = fx_s{i,2,n}; fxR_s{i,1,n} = fx_n{i,d2,n}; fxR_e{i,1,n} = fx_w{i+1,1,n}; fxR_w{i,1,n} = fx_e{i-1,1,n};
            fyR_n{i,1,n} = fy_s{i,2,n}; fyR_s{i,1,n} = fy_n{i,d2,n}; fyR_e{i,1,n} = fy_w{i+1,1,n}; fyR_w{i,1,n} = fy_e{i-1,1,n};
        end
    end

% Eastmost and Westmost cells
    for j=2:d2-1
        for n=1:neq
% Eastmost
            uR_n{d1,j,n} = u_s{d1,j+1,n}; uR_s{d1,j,n} = u_n{d1,j-1,n}; uR_e{d1,j,n} = u_w{1,j,n}; uR_w{d1,j,n} = u_e{d1-1,j,n};
            fxR_n{d1,j,n} = fx_s{d1,j+1,n}; fxR_s{d1,j,n} = fx_n{d1,j-1,n}; fxR_e{d1,j,n} = fx_w{1,j,n}; fxR_w{d1,j,n} = fx_e{d1-1,j,n};
            fyR_n{d1,j,n} = fy_s{d1,j+1,n}; fyR_s{d1,j,n} = fy_n{d1,j-1,n}; fyR_e{d1,j,n} = fy_w{1,j,n}; fyR_w{d1,j,n} = fy_e{d1-1,j,n};

% Westmost
            uR_n{1,j,n} = u_s{1,j+1,n}; uR_s{1,j,n} = u_n{1,j-1,n}; uR_e{1,j,n} = u_w{2,j,n}; uR_w{1,j,n} = u_e{d1,j,n};
            fxR_n{1,j,n} = fx_s{1,j+1,n}; fxR_s{1,j,n} = fx_n{1,j-1,n}; fxR_e{1,j,n} = fx_w{2,j,n}; fxR_w{1,j,n} = fx_e{d1,j,n};
            fyR_n{1,j,n} = fy_s{1,j+1,n}; fyR_s{1,j,n} = fy_n{1,j-1,n}; fyR_e{1,j,n} = fy_w{2,j,n}; fyR_w{1,j,n} = fy_e{d1,j,n};
        end
    end

    for n=1:neq
% NE
        uR_n{d1,d2,n} = u_s{d1,1,n}; uR_s{d1,d2,n} = u_n{d1,d2-1,n}; uR_e{d1,d2,n} = u_w{1,d2,n}; uR_w{d1,d2,n} = u_e{d1-1,d2,n};
        fxR_n{d1,d2,n} = fx_s{d1,1,n}; fxR_s{d1,d2,n} = fx_n{d1,d2-1,n}; fxR_e{d1,d2,n} = fx_w{1,d2,n}; fxR_w{d1,d2,n} = fx_e{d1-1,d2,n};
        fyR_n{d1,d2,n} = fy_s{d1,1,n}; fyR_s{d1,d2,n} = fy_n{d1,d2-1,n}; fyR_e{d1,d2,n} = fy_w{1,d2,n}; fyR_w{d1,d2,n} = fy_e{d1-1,d2,n};
% NW
        uR_n{1,d2,n} = u_s{1,1,n}; uR_s{1,d2,n} = u_n{1,d2-1,n}; uR_e{1,d2,n} = u_w{2,d2,n}; uR_w{1,d2,n} = u_e{d1,d2,n};
        fxR_n{1,d2,n} = fx_s{1,1,n}; fxR_s{1,d2,n} = fx_n{1,d2-1,n}; fxR_e{1,d2,n} = fx_w{2,d2,n}; fxR_w{1,d2,n} = fx_e{d1,d2,n};
        fyR_n{1,d2,n} = fy_s{1,1,n}; fyR_s{1,d2,n} = fy_n{1,d2-1,n}; fyR_e{1,d2,n} = fy_w{2,d2,n}; fyR_w{1,d2,n} = fy_e{d1,d2,n};
% SE
        uR_n{d1,1,n} = u_s{d1,2,n}; uR_s{d1,1,n} = u_n{d1,d2,n}; uR_e{d1,1,n} = u_w{1,1,n}; uR_w{d1,1,n} = u_e{d1-1,1,n};
        fxR_n{d1,1,n} = fx_s{d1,2,n}; fxR_s{d1,1,n} = fx_n{d1,d2,n}; fxR_e{d1,1,n} = fx_w{1,1,n}; fxR_w{d1,1,n} = fx_e{d1-1,1,n};
        fyR_n{d1,1,n} = fy_s{d1,2,n}; fyR_s{d1,1,n} = fy_n{d1,d2,n}; fyR_e{d1,1,n} = fy_w{1,1,n}; fyR_w{d1,1,n} = fy_e{d1-1,1,n};
% SW
        uR_n{1,1,n} = u_s{1,2,n}; uR_s{1,1,n} = u_n{1,d2,n}; uR_e{1,1,n} = u_w{2,1,n}; uR_w{1,1,n} = u_e{d1,1,n};
        fxR_n{1,1,n} = fx_s{1,2,n}; fxR_s{1,1,n} = fx_n{1,d2,n}; fxR_e{1,1,n} = fx_w{2,1,n}; fxR_w{1,1,n} = fx_e{d1,1,n};
        fyR_n{1,1,n} = fy_s{1,2,n}; fyR_s{1,1,n} = fy_n{1,d2,n}; fyR_e{1,1,n} = fy_w{2,1,n}; fyR_w{1,1,n} = fy_e{d1,1,n};
    end

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
                    flux_n{i,j,n} =  1/2*(fy_n{i,j,n}+fyR_n{i,j,n}) - alpha/2 * (uR_n{i,j,n} - u_n{i,j,n});
                    flux_s{i,j,n} = -1/2*(fy_s{i,j,n}+fyR_s{i,j,n}) - alpha/2 * (uR_s{i,j,n} - u_s{i,j,n});
                    flux_e{i,j,n} =  1/2*(fx_e{i,j,n}+fxR_e{i,j,n}) - alpha/2 * (uR_e{i,j,n} - u_e{i,j,n});
                    flux_w{i,j,n} = -1/2*(fx_w{i,j,n}+fxR_w{i,j,n}) - alpha/2 * (uR_w{i,j,n} - u_w{i,j,n});
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
                beta_y=-2*pi*radius/(12*86400)*sin_angle*sin(qp_x);     % Vector
                beta_y_e=-2*pi*radius/(12*86400)*sin_angle*sin(qp_x_e); % Scalar
                beta_y_w=-2*pi*radius/(12*86400)*sin_angle*sin(qp_x_w); % Scalar
                alpha_n = max( sqrt(beta_x_n.^2+beta_y.^2), [], 1); alpha_s = max( sqrt(beta_x_s.^2+beta_y.^2), [], 1);
                alpha_e = max( sqrt(beta_x.^2+beta_y_e^2), [], 1); alpha_w = max( sqrt(beta_x.^2+beta_y_w^2), [], 1);

		fact_bd = cos(qp_y);
% TODO: confirm the signs for each face (5.3.20: looks correct to me)
% TODO: check the fact_bd factor...  Why is this not simply a scalar???  Why does it not apply to EW fluxes??
                for n=1:neq
                    flux_n{i,j,n} =  1/2*(fy_n{i,j,n}+fyR_n{i,j,n}).*fact_bd - alpha_n/2 * (uR_n{i,j,n} - u_n{i,j,n}) .* fact_bd; 
                    flux_s{i,j,n} = -1/2*(fy_s{i,j,n}+fyR_s{i,j,n}).*fact_bd - alpha_s/2 * (uR_s{i,j,n} - u_s{i,j,n}) .* fact_bd;
                    flux_e{i,j,n} =  1/2*(fx_e{i,j,n}+fxR_e{i,j,n}) - alpha_e/2 * (uR_e{i,j,n} - u_e{i,j,n});
                    flux_w{i,j,n} = -1/2*(fx_w{i,j,n}+fxR_w{i,j,n}) - alpha_w/2 * (uR_w{i,j,n} - u_w{i,j,n});
                end
            end
	end

    case "swe_sphere"

% Not yet implemented.

end

end
