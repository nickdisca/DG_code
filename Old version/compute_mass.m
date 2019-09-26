function [mass] =compute_mass(phi,r,wts2d,d1,d2,hx,hy,factor)

dim=(r+1)^2;
mass=NaN*zeros(dim,dim,d1*d2);
determ=hx*hy/4;

for k=1:d1*d2
    for i=1:dim
        for j=1:dim
            mass(i,j,k)=determ*wts2d'*(phi(:,i).*phi(:,j).*factor(:,k));
        end
    end
end

end