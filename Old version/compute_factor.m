function [fact_int,fact_bd,complem_fact,radius] = compute_factor(eq_type,r,d1,d2,y_qp,y_qp_bd)

if eq_type=="linear" || eq_type=="swe"
    fact_int=ones(size(y_qp)); 
    fact_bd=ones(size(y_qp_bd)); 
    complem_fact=zeros(size(y_qp));
    radius=1;
end

if eq_type=="sphere" || eq_type=="swe_sphere"
    
    fact_int=cos(y_qp);
    fact_bd=cos(y_qp_bd);
    complem_fact=sin(y_qp);
    %fact_bd(:,d2:d2:d1*d2,3)=0; fact_bd(:,1:d2:(d1-1)*d2+1,1,:)=0; %force factor to be zero at poles
    radius=6.37122e6;
    
end

end
