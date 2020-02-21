function [flux_x, flux_y] = flux_function_new(u,eq_type,radius,hx,hy,x_c,y_c,pts2d_x,pts2d_y)

d1=size(u,1);
d2=size(u,2);
flux_x=cell(d1,d2);
flux_y=cell(d1,d2);

switch eq_type
    
    case "linear"
        for i=1:d1
            for j=1:d2
                flux_x{i,j}=u{i,j};
                flux_y{i,j}=u{i,j};
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
                qp_x=x_c(i)+pts2d_x/2*hx;
                qp_y=y_c(j)+pts2d_y/2*hy;
                beta_x=2*pi*radius/(12*86400)*(cos(qp_y)*cos(angle)+sin(qp_y).*cos(qp_x)*sin(angle));
                beta_y=-2*pi*radius/(12*86400)*sin(angle)*sin(qp_x);
                flux_x{i,j}=beta_x.*u{i,j};
                flux_y{i,j}=beta_y.*u{i,j};
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
