function [x2d,w2d] = tensor_product(x1d,y1d,w1d)
%assemble 2D points and weights using tensor product argument, all in the
%reference element [-1,1]
%the points are assembled with in columnwise fashion, from bottom to top

dim=length(x1d);

x2d=nan(dim^2,2);
for i=1:dim^2
    x2d(i,2)=y1d(mod(i-1,dim)+1);
    x2d(i,1)=x1d(floor((i-1)/dim)+1);
end

w2d=NaN*zeros(dim^2,1);
for i=1:dim^2
    w2d(i)=w1d(mod(i-1,dim)+1)*w1d(floor((i-1)/dim)+1);
end
    
return