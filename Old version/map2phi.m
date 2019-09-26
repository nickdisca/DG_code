function [x_phi] = map2phi(x,r,x_e,y_e,d1,d2,hx,hy)

x_phi=NaN*zeros(2*(r+1)^2,d1*d2);
x=x(:);

for i=1:d1
    for j=1:d2
        x_phi(1:(r+1)^2,(i-1)*d2+j)=x_e(i)+(1+x(1:(r+1)^2))/2*hx;
        x_phi((r+1)^2+1:2*(r+1)^2,(i-1)*d2+j)=y_e(j)+(1+x((r+1)^2+1:2*(r+1)^2))/2*hy;
    end
end

end