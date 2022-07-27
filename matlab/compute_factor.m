function [fact_int,fact_bd,complem_fact,radius] = compute_factor(eq_type,y_qp,y_qp_bd)
%compute metric factor due to integration over sphere, i.e. cosine of the latitude. 
%This is needed for both internal and boundary integrals. 
%A complementary factor is needed for SWE on the sphere (only internal ints).

%cartesian geometry
if eq_type=="linear" || eq_type=="swe"
    fact_int=ones(size(y_qp)); 
    fact_bd=ones(size(y_qp_bd)); 
    complem_fact=zeros(size(y_qp));
    radius=1;
end

%spherical geometry
if eq_type=="adv_sphere" || eq_type=="swe_sphere"    
    fact_int=cos(y_qp);
    fact_bd=cos(y_qp_bd);
    complem_fact=sin(y_qp);
    radius=6.37122e6;
%     radius=1;
end

end
