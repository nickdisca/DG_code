function [x_phi] = map2phi_static(x,r,x_e,y_e,d1,d2,hx,hy)
%given points in the reference element, maps them in the physical domain,
%and returns a matrix containing the x and y coordinates stacked

%dimension: (number_of_points_x+number_of_points_y) x (num_elements)
x_phi=NaN*zeros(2*(r+1)^2,d1*d2);

for i=1:d1
    for j=1:d2
        
        %x coordinates
        x_phi(1:(r+1)^2,(i-1)*d2+j)=x_e(i)+(1+x(:,1))/2*hx;
        
        %y coordinates
        x_phi((r+1)^2+1:end,(i-1)*d2+j)=y_e(j)+(1+x(:,2))/2*hy;
        
    end
end

end