function [flux_x, flux_y] = flux_function(u,eq_type,radius,hx,hy,x_c,y_c,pts_x,pts_y)
%
% This is a generic function for evaluating flux vector f at the points described by [pts_x,pts_y].
% It can be called inside the cell:   pts_x = pts2d_x and pts_y = pts2d_y
% Or on the boundary, where [x_c,y_c] is the edge center, 
%     NS:  pts_x = pts and pts_y = zeros(size(pts)) 
%     EW:  pts_x = zeros(size(pts)) and pts_y = pts
%
d1=size(u,1);
d2=size(u,2);
neq=size(u,3);
flux_x=cell(size(u));
flux_y=cell(size(u));

switch eq_type
    
    case "linear"
        for i=1:d1
            for j=1:d2
                for n=1:neq
                    flux_x{i,j,n}=u{i,j,n};
                    flux_y{i,j,n}=u{i,j,n};
                end
            end
        end

    case "swe"
%
% Not currently supported
%        g=9.80616;
%        flux_x(:,:,1)=u(:,:,2);
%        flux_y(:,:,1)=u(:,:,3);
%        flux_x(:,:,2)=u(:,:,2).^2./u(:,:,1)+g/2*u(:,:,1).^2;
%        flux_y(:,:,2)=u(:,:,2).*u(:,:,3)./u(:,:,1);
%        flux_x(:,:,3)=u(:,:,2).*u(:,:,3)./u(:,:,1);
%        flux_y(:,:,3)=u(:,:,3).^2./u(:,:,1)+g/2*u(:,:,1).^2;
        
    case "adv_sphere"
%
% Here we have to create pts2d_phi_x (qp_x) and pts2d_phi_y (qp_y) on the fly, see map2phi_static
%
        angle=pi/2;
        
        for i=1:d1
            for j=1:d2
                for n=1:neq
                    qp_x=x_c(i)+pts_x*hx/2;
                    qp_y=y_c(j)+pts_y*hy/2;
                    beta_x=2*pi*radius/(12*86400)*(cos(qp_y)*cos(angle)+sin(qp_y).*cos(qp_x)*sin(angle));
                    beta_y=-2*pi*radius/(12*86400)*sin(angle)*sin(qp_x);
                    flux_x{i,j,n}=beta_x.*u{i,j,n};
                    flux_y{i,j,n}=beta_y.*u{i,j,n};
                end
            end
        end

    case "swe_sphere"
%
% Not currently supported
%        g=9.80616;
%        flux_x(:,:,1)=u(:,:,2);
%        flux_y(:,:,1)=u(:,:,3);
%        flux_x(:,:,2)=u(:,:,2).^2./u(:,:,1)+g/2*u(:,:,1).^2;
%        flux_y(:,:,2)=u(:,:,2).*u(:,:,3)./u(:,:,1);
%        flux_x(:,:,3)=u(:,:,2).*u(:,:,3)./u(:,:,1);
%        flux_y(:,:,3)=u(:,:,3).^2./u(:,:,1)+g/2*u(:,:,1).^2;        
        
end

end
