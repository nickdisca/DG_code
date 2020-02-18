function [x_phi,y_phi] = map2phi_static_new(x,y,n_qp,x_e,y_e,d1,d2,hx,hy)
%given points in the reference element, maps them in the physical domain,
%and returns a matrix containing the x and y coordinates stacked

%dimension: (number_of_points_x+number_of_points_y) x (num_elements)
x_phi=cell(d1,d2);
y_phi=cell(d1,d2);

for i=1:d1
    for j=1:d2

        x_phi{i,j} = zeros(n_qp,1);
        y_phi{i,j} = zeros(n_qp,1);
        %x coordinates
        x_phi{i,j}=x_e(i)+(1+x(:))/2*hx;
        
        %y coordinates
        y_phi{i,j}=y_e(j)+(1+y(:))/2*hy;
        
    end
end

end
