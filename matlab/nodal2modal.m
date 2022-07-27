function uM=nodal2modal(uN,V,r)
%convert nodal to modal using the Vandermonde matrix V and variable degree
%r

d1=size(r,2);
d2=size(r,1);
uM=zeros(size(uN));

for n=1:size(uN,3)
    for i=1:d1
        for j=1:d2
            uM(1:(r(j,i)+1)^2,(i-1)*d2+j, n)=V{r(j,i)}\uN(1:(r(j,i)+1)^2,(i-1)*d2+j, n);
        end
    end
end

end