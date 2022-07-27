function [mass,inv_mass] =compute_mass(phi,wts2d,d1,d2,r,hx,hy,factor)
%compute mass matrix and its inverse, different in all elements due to cosine factors
%M(i,j)=integral_K(Phi_j*Phi_i*dA)

dim=(max(r(:))+1)^2;

% TODO: shouldn't this be zeros?
%dimensions: (cardinality)x(cardinality)x(num_elems)
mass=repmat(eye(dim,dim),1,1,d1*d2);
inv_mass=repmat(eye(dim,dim),1,1,d1*d2);


% mass=repmat(zeros(dim,dim),1,1,d1*d2);
% inv_mass=repmat(zeros(dim,dim),1,1,d1*d2);



%determinant of the affine mapping from reference to physical element (this
%is assumed to be constant)
determ=hx*hy/4;

for k=1:d1*d2
    
    elem_x=floor((k-1)/d2)+1; % element x index
    elem_y=mod((k-1),d2)+1; % element y index
    r_loc=r(elem_y,elem_x);
    
    for i=1:(r_loc+1)^2
        for j=1:(r_loc+1)^2
            %det*sum(i-th basis function in qp * j-th basis function in qp * metric
            %factor * weights)
            mass(i,j,k)=determ*wts2d'*(phi{r_loc}(:,i).*phi{r_loc}(:,j).*factor(:,k));
%             mass(i,j,k)=determ*wts2d'*(phi{r_loc}(:,i).*phi{r_loc}(:,j));

        end
    end
    
    inv_mass(1:(r_loc+1)^2,1:(r_loc+1)^2,k)=mass(1:(r_loc+1)^2,1:(r_loc+1)^2,k)\eye((r_loc+1).^2);
    
end

end