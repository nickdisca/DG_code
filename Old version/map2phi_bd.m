function [x_phi] = map2phi_bd(x,r,x_e,y_e,d1,d2,hx,hy)

x_phi=NaN*zeros(2*(r+1),d1*d2,4);
x=x(:);

for i=1:d1
    for j=1:d2
        for k=1:4
            x_phi(1:(r+1),(i-1)*d1+j,k)=x_e(i)+(1+x)/2*hx;
            x_phi((r+1)+1:2*(r+1),(i-1)*d1+j,k)=y_e(j)+(1+x)/2*hy;
            
            if k==1, x_phi((r+1)+1:2*(r+1),(i-1)*d1+j,k)=y_e(j); end
            if k==2, x_phi(1:(r+1),(i-1)*d1+j,k)=x_e(i)+hx; end
            if k==3, x_phi((r+1)+1:2*(r+1),(i-1)*d1+j,k)=y_e(j)+hy; end
            if k==4, x_phi(1:(r+1),(i-1)*d1+j,k)=x_e(i); end
            
        end
    end
end

end