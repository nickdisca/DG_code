function [alpha] = get_maximum_eig(u,eq_type,radius,qp_x,qp_y)
%compute maximum wave speed over the given face

%dimensions: (num_bd_qp)x(num_elems)x(num_faces), with the same value for
%all quadrature points
alpha=nan(size(u,1),size(u,2),size(u,3));

switch eq_type
    
    case "linear"
        alpha=ones(size(u,1),size(u,2),size(u,3));
        
    case "swe"
        g=9.80616;
        alpha_tmp = max( sqrt(abs(g*u(:,:,:,1)))+sqrt((u(:,:,:,2)./u(:,:,:,1)).^2+(u(:,:,:,3)./u(:,:,:,1)).^2),[],1);
        if min(min(min(u(:,:,:,1))))<0
            warning('Positivity is lost: h=%f!',min(min(min(u(:,:,:,1)))));
        end
        for i=1:size(u,1)
            alpha(i,:,:)=alpha_tmp;
        end
        
    case "adv_sphere"
        angle=pi/2;
        beta_x=2*pi*radius/(12*86400)*(cos(qp_y)*cos(angle)+sin(qp_y).*cos(qp_x)*sin(angle));
        beta_y=-2*pi*radius/(12*86400)*sin(angle)*sin(qp_x);
        alpha_tmp = max( sqrt(beta_x.^2+beta_y.^2), [], 1);
        for i=1:size(u,1)
            alpha(i,:,:)=alpha_tmp;
        end  
        
    case "swe_sphere"
        g=9.80616;
        alpha_tmp = max( sqrt(abs(g*u(:,:,:,1)))+sqrt((u(:,:,:,2)./u(:,:,:,1)).^2+(u(:,:,:,3)./u(:,:,:,1)).^2),[],1);
        if min(min(min(u(:,:,:,1))))<0
            warning('Positivity is lost: h=%f!',min(min(min(u(:,:,:,1)))));
        end
        for i=1:size(u,1)
            alpha(i,:,:)=alpha_tmp;
        end        
        
end

end