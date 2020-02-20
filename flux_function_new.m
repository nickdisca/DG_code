function [flux_x, flux_y] = flux_function(u,eq_type,radius,pts2d_x,pts2d_y)

flux_x=cell(size(u));

switch eq_type
    
    case "linear"
        for i=1:d1
            for j=1:d2
                flux_x{i,j}=u{i,j};
                flux_y{i,j}=u{i,j};

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
        beta_x=2*pi*radius/(12*86400)*(cos(qp_y)*cos(angle)+sin(qp_y).*cos(qp_x)*sin(angle));
        beta_y=-2*pi*radius/(12*86400)*sin(angle)*sin(qp_x);
        flux_x=beta_x.*u; 
        flux_y=beta_y.*u;

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
