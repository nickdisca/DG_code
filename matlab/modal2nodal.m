function uN=modal2nodal(uM,V,r)
%convert modal to nodal using the Vandermonde matrix V and variable degree
%r. The matrix V can be rectangular
d1=size(r,2);
d2=size(r,1);
uN=zeros(size(uM));

for n=1:size(uM,3)
    for i=1:d1
        for j=1:d2
            uN(1:size(V{r(j,i)},1),(i-1)*d2+j, n)=V{r(j,i)}*uM(1:(r(j,i)+1)^2,(i-1)*d2+j,n);
        end
    end
end

end