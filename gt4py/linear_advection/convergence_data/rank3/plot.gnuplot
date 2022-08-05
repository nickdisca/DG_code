set terminal qt size 700,500 enhanced font ', 16'
unset grid
set title 'Runge-Kutta 3'
set xlabel 'Mesh Size'
set ylabel '||u(x,y,1) - u_0(x,y)||_{L^2}'
set logscale
set key bottom right Left reverse box lw 0.2 spacing 1.4
set xrange [0.00625:0.05]
set format y "%g"
f(x) = 0.15*x
g(x) = 1.5*x**3
p 'space_3_time_1.csv' u 2:3 w lp t 'p=0',\
  'space_3_time_2.csv' u 2:3 w lp t 'p=1', \
  'space_3_time_3.csv' u 2:3 w lp t 'p=2', \
  'space_3_time_4.csv' u 2:3 w lp t 'p=3', \
  f(x) dashtype 2 lw 2 lc 'gray' t 'O(h)', \
  g(x) dashtype 3 lw 2 lc 'gray' t 'O(h^3)'
