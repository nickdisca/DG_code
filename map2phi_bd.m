function [x_phi] = map2phi_bd(x,r,x_e,y_e,d1,d2,hx,hy)
%given points in the reference element, maps them on the boundary of all elements,
%and returns an array containing the x and y coordinates stacked, for all
%faces of the element

%convention: 1=bottom, 2=right, 3=top, 4=left

%dimension: (num_x_points+num_y_points)x(num_elements)x(num_faces)
x_phi=nan(2*(r+1),d1*d2,4);

for i=1:d1
    for j=1:d2
        
        %loop over faces
        for k=1:4
            
            %x coordinates
            x_phi(1:(r+1),(i-1)*d1+j,k)=x_e(i)+(1+x)/2*hx;
            %y coordinates
            x_phi((r+1)+1:2*(r+1),(i-1)*d1+j,k)=y_e(j)+(1+x)/2*hy;
            
            %correct based on the face where I am sitting
            switch k
                %bottom -> fix y coordinate
                case 1
                    x_phi((r+1)+1:2*(r+1),(i-1)*d1+j,k)=y_e(j)*ones(r+1,1);
                    
                %right -> fix x coordinate
                case 2
                    x_phi(1:(r+1),(i-1)*d1+j,k)=(x_e(i)+hx)*ones(r+1,1); 
                    
                %top -> fix y coordinate
                case 3
                    x_phi((r+1)+1:2*(r+1),(i-1)*d1+j,k)=(y_e(j)+hy)*ones(r+1,1);
                
                %left -> fix x coordinate
                case 4
                    x_phi(1:(r+1),(i-1)*d1+j,k)=x_e(i)*ones(r+1,1);
            end
            
        end
    end
end

end