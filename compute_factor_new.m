function [fact_int,fact_bd,complem_fact,radius] = compute_factor_new(d1,d2,eq_type,y_qp,y_qp_bd)
%compute metric factor due to integration over sphere, i.e. cosine of the latitude. 
%This is needed for both internal and boundary integrals. 
%A complementary factor is needed for SWE on the sphere (only internal ints).

fact_int=cell(d1,d2);
complem_fact=cell(d1,d2);
fact_bd=cell(d1,d2,4);

for i=1:d1
    for j=1:d2

%cartesian geometry
        if eq_type=="linear" || eq_type=="swe"
            fact_int=ones(size(y_qp{i,j})); 
            fact_bd=ones(size(y_qp_bd{i,j})); 
            complem_fact=zeros(size(y_qp{i,j}));
            radius=1;
        end

%spherical geometry
        if eq_type=="adv_sphere" || eq_type=="swe_sphere"    
            fact_int=cos(y_qp{i,j});
            fact_bd=cos(y_qp_bd{i,j});
            complem_fact=sin(y_qp{i,j});
            radius=6.37122e6;
        end
    end
end

end
