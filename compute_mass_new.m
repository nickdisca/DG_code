function [mass,inv_mass] =compute_mass_new(phi,wts2d,d1,d2,r,hx,hy,factor)
%compute mass matrix and its inverse, different in all elements due to cosine factors
%M(i,j)=integral_K(Phi_j*Phi_i*dA)

dim=(max(r(:))+1)^2;

%dimensions: cell arrays (d1,d2)
mass=cell(d1,d2);
inv_mass=cell(d1,d2);

%determinant of the affine mapping from reference to physical element (this
%is assumed to be constant)
determ=hx*hy/4;

for i=1:d1
    for j=1:d2
        r_loc=r(i,j);
        mass{i,j} = zeros((rloc+1)^2);
        for m=1:(r_loc+1)^2
            for n=1:(r_loc+1)^2
                %det*sum(i-th basis function in qp * j-th basis function in qp * metric
                %factor * weights)
                mass{i,j}(m,n)=determ*wts2d'*(phi{r_loc}(:,m).*phi{r_loc}(:,n).*factor{i,j});
            end
        end
        inv_mass{i,j}=inv(mass{i,j});
    end
end
