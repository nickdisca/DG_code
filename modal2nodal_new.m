function uN=modal2nodal_new(uM,V,r)
%convert modal to nodal using the Vandermonde matrix V and variable degree
%r. The matrix V can be rectangular
d1=size(uM,1);
d2=size(uM,2);
uN=cell(d1,d2);

for i=1:d1
    for j=1:d2
	size_V = size(V{r(i,j)},1);
        uN{i,j}=V{r(i,j)}*uM{i,j};
    end
end

end
