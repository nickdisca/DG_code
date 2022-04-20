function [r] = degree_distribution(type,d1,d2,r_max)
%compute spatial distribution of the degrees

switch type
    %uniform distribution
    case "unif"
        r=r_max*ones(d2,d1);
    
    %maximum degree close to the central part of domain, only y variability
    case "y_dep"
        r_vec=round( (r_max-1)/(floor(d2/2)-1)*(0:floor(d2/2)-1)+1 );
        r_vec=[r_vec fliplr(r_vec)];
        if mod(d2,2)==1
            r_vec=[r_vec(1:end/2) r_vec(end/2) r_vec(end/2+1:end)];
        end
        r=repmat(r_vec',1,d1);
        
    %monotonically increasing degree, only y variablity    
    case "y_incr"
        r_vec=round ( (r_max-1)/(d2-1)*(0:d2-1)+1 );
        r=repmat(r_vec',1,d1);
        
    otherwise
        error('Degree distribution not assigned');

end

r(r>r_max)=r_max;

end