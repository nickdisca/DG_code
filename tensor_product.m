function [x2d,w2d] = tensor_product(x1d,y1d,w1d)

x2d=NaN*zeros(size(x1d,1)^2,2);
for i=1:size(x1d,1)^2
    x2d(i,2)=y1d(mod(i-1,size(x1d,1))+1);
    x2d(i,1)=x1d(floor((i-1)/size(x1d,1))+1);
end

w2d=NaN*zeros(length(w1d)^2,1);
for i=1:length(w1d)^2
    w2d(i)=w1d(mod(i-1,length(w1d))+1)*w1d(floor((i-1)/length(w1d))+1);
end
    
return