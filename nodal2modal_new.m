function uM=nodal2modal_new(uN,V,r)
%convert nodal to modal using the Vandermonde matrix V and variable degree r

d1=size(uN,1);
d2=size(uN,2);
uM=cell(d1,d2);

for i=1:d1
    for j=1:d2
        uM{i,j} = (V{r(i,j)}\uN{i,j});
    end
end

end
