set terminal qt enhanced size 700,500 font ',16'
set title 'Absolute Error vs Mesh Size'
set xlabel 'Mesh Size'
set ylabel '||u(x,y,1) - u_0(x,y)||_{L^2}'
set logscale
set xrange [0.00625:0.05]
set key bottom right Left reverse box
p 'space_1_time_1.csv' u 2:3 w lp t 'p=0 ',\
  'space_2_time_2.csv' u 2:3 w lp t 'p=1 ', \
  'space_3_time_3.csv' u 2:3 w lp t 'p=2 ', \
  'space_4_time_4.csv' u 2:3 w lp t 'p=3 ', \
