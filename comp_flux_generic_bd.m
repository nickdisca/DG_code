function [flux] = comp_flux_generic_bd(u,off_x,off_y,pts,d1,d2,neq,hx,hy,eq_type,radius,x_c,y_c)

% Attempt to determine fluxes on boundary for any of the 4 faces
%
% off_x, off_y : can be in {-1,0,1}    N: [0,1], S: [0,-1], E: [1,0], W: [-1,0]

flux = cell(d1,d2,neq);

%
% Find the U values from the N, S, E, W neighboring cells
%         

pts_x = off_x*ones(size(pts))+abs(off_y)*pts;  % Place coordinates in the right place
pts_y = off_y*ones(size(pts))+abs(off_x)*pts;  % with respect to the center point

[fx, fy] = flux_function_new(u,eq_type,radius,hx,hy,x_c,y_c,pts_x,pts_y);

pts_x = -off_x*ones(size(pts))+abs(off_y)*pts;  % place coordinates of opposing face
pts_y = -off_y*ones(size(pts))+abs(off_x)*pts;  % with respect to the center point

[fRx, fRy] = flux_function_new(u,eq_type,radius,hx,hy,x_c,y_c,pts_x,pts_y);
% fx_7_5 = fx{7,5,1}'
% fy_7_5 = fy{7,5,1}'

switch eq_type

    case "linear"

        alpha = 1.0;
        for i=1:d1
            for j=1:d2
                in=mod(i-1+off_x,d1)+1; jn=mod(j-1+off_y,d2)+1;   % Find the index of neighbor sharing this edge 
                for n=1:neq
                    flux{i,j,n} = 1/2*(off_x*(fx{i,j,n}+fRx{in,jn,n}) + off_y*(fy{i,j,n}+fRy{in,jn,n})) - alpha/2*(u{in,jn,n}-u{i,j,n});
                end
            end
        end
%
    case "swe"

        g=9.80616;
        for i=1:d1
            for j=1:d2
                in = mod(i-1+off_x,d1)+1; jn=mod(j-1+off_y,d2)+1;   % Find the index of neighbor sharing this edge  -- wrap around
                % Calculate the maximum wave speed over the face (Scalar: same for all QP)
                alphaL =  max( sqrt(abs(g*u{i,j,1}))+sqrt((u{i,j,2}./u{i,j,1}).^2+(u{i,j,3}./u{i,j,1}).^2) );           % Scalar
                alphaR =  max( sqrt(abs(g*u{in,jn,1}))+sqrt((u{in,jn,2}./u{in,jn,1}).^2+(u{in,jn,3}./u{in,jn,1}).^2) ); % Scalar
                alpha  =  max(alphaL, alphaR);
                for n=1:neq
                    flux{i,j,n} = 1/2*(off_x*(fx{i,j,n}+fRx{in,jn,n}) + off_y*(fy{i,j,n}+fRy{in,jn,n})) - alpha/2*(u{in,jn,n}-u{i,j,n});
                end

            end
        end

    case "adv_sphere"
        angle = pi/2;
        cos_angle = 0.0;
	sin_angle = 1.0;
        for i=1:d1
            for j=1:d2
                in = mod(i-1+off_x,d1)+1; jn=mod(j-1+off_y,d2)+1;   % Find the index of neighbor sharing this edge  -- wrap around
                qp_x=x_c(i)+abs(off_y)*pts*hx/2 + off_x*hx/2;       % Vector of x-indices of boundary quadrature points
                qp_y=y_c(j)+abs(off_x)*pts*hy/2 + off_y*hy/2;       % Vector of y-indices of boundary quadrature points
                beta_x=2*pi*radius/(12*86400)*(cos(qp_y)*cos_angle+sin(qp_y).*cos(qp_x)*sin_angle);      % Vector
                beta_y=-2*pi*radius/(12*86400)*sin_angle*sin(qp_x);                                      % Vector
                alpha = max( sqrt(beta_x.^2+beta_y.^2), [], 1);                                          % Scalar

                fact_bd = cos(qp_y);
                if ( i==7 && j==5 )
                   R_7_5 = [u{in,jn,n}' fx{in,jn,n}' fy{in,jn,n}']
%%%                    fxyLmfxyR_7_5 = [fx{i,j,n}' fx{in,jn,n}' fy{i,j,n}' fy{in,jn,n}']
%%% SAME                    fact_bd_7_5 = [i, j, in, jn, fact_bd']
                end
% TODO: confirm the signs for each face (5.3.20: looks correct to me)
% TODO: check the fact_bd factor...  Why is this not simply a scalar???  Why does it not apply to EW fluxes??
                for n=1:neq
                    flux{i,j,n} = 1/2*(off_x*(fx{i,j,n}+fRx{in,jn,n}) + off_y*(fy{i,j,n}+fRy{in,jn,n}).*fact_bd)...
                                  - alpha/2*(u{in,jn,n}-u{i,j,n}).*(abs(off_x)+fact_bd*abs(off_y));
                end
            end
	end

    case "swe_sphere"

% Not yet implemented.

end

end
