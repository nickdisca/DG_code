function [flux_x, flux_y] = flux_function(u,eq_type,radius,qp_x,qp_y)

sz_u=size(u);
is_boundary=(length(sz_u)==4);

if is_boundary
    u=reshape(u,sz_u(1),sz_u(2)*sz_u(3),sz_u(4));
    qp_x=reshape(qp_x,sz_u(1),sz_u(2)*sz_u(3));
    qp_y=reshape(qp_y,sz_u(1),sz_u(2)*sz_u(3));
end

switch eq_type
    
    case "linear"
        flux_x=u;
        flux_y=u;

    case "swe"
        g=9.80616;
        flux_x(:,:,1)=u(:,:,2);
        flux_y(:,:,1)=u(:,:,3);
        flux_x(:,:,2)=u(:,:,2).^2./u(:,:,1)+g/2*u(:,:,1).^2;
        flux_y(:,:,2)=u(:,:,2).*u(:,:,3)./u(:,:,1);
        flux_x(:,:,3)=u(:,:,2).*u(:,:,3)./u(:,:,1);
        flux_y(:,:,3)=u(:,:,3).^2./u(:,:,1)+g/2*u(:,:,1).^2;
        
    case "adv_sphere"
        angle=pi/2;
        beta_x=2*pi*radius/(12*86400)*(cos(qp_y)*cos(angle)+sin(qp_y).*cos(qp_x)*sin(angle));
        beta_y=-2*pi*radius/(12*86400)*sin(angle)*sin(qp_x);
        flux_x=beta_x.*u; 
        flux_y=beta_y.*u;

    case "swe_sphere"
        g=9.80616;
        flux_x(:,:,1)=u(:,:,2);
        flux_y(:,:,1)=u(:,:,3);
        flux_x(:,:,2)=u(:,:,2).^2./u(:,:,1)+g/2*u(:,:,1).^2;
        flux_y(:,:,2)=u(:,:,2).*u(:,:,3)./u(:,:,1);
        flux_x(:,:,3)=u(:,:,2).*u(:,:,3)./u(:,:,1);
        flux_y(:,:,3)=u(:,:,3).^2./u(:,:,1)+g/2*u(:,:,1).^2;        
        
end

if is_boundary
    flux_x=reshape(flux_x,sz_u(1),sz_u(2),sz_u(3),sz_u(4));
    flux_y=reshape(flux_y,sz_u(1),sz_u(2),sz_u(3),sz_u(4));
end

end