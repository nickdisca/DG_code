function [X]=convert_cell_2_global(x,r,n_qp,dim,d1,d2)
%convert x in a cell format to a cell array containing global matrices

%indexes to assemble the matrix
idx_r=repelem(reshape((1:n_qp*d1*d2),n_qp,d1*d2),1,dim);
idx_c=repmat(1:dim*d1*d2,n_qp,1);

%dimensions: (n_qp)x(cardinality)x(n_elem)
X_tensor=zeros(n_qp,dim,d1*d2);

for k=1:size(x{1},3)
    for i=1:d1
        for j=1:d2
            sz=size(x{r(j,i)});
            X_tensor(1:sz(1),1:sz(2),(i-1)*d2+j)=x{r(j,i)}(:,:,k);
        end
    end
    X{k}=sparse(idx_r(:),idx_c(:),X_tensor(:));
end

