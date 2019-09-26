function [mass] =compute_mass(phi,wts2d,d1,d2,hx,hy,factor)
%compute mass matrix, different in all elements due to cosine factors
%M(i,j)=integral_K(Phi_j*Phi_i*dA)

dim=size(phi,2);

%dimensions: (cardinality)x(cardinality)x(num_elems)
mass=nan(dim,dim,d1*d2);

%determinant of the affine mapping from reference to physical element (this
%is assumed to be constant)
determ=hx*hy/4;

for k=1:d1*d2
    
    for i=1:dim
        for j=1:dim
            %det*sum(i-th basis function in qp * j-th basis function in qp * metric
            %factor * weights)
            mass(i,j,k)=determ*wts2d'*(phi(:,i).*phi(:,j).*factor(:,k));
        end
    end
    
end

end