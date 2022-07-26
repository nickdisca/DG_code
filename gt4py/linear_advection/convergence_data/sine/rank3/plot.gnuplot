set terminal qt enhanced font ',20'
unset grid
set title 'Absolute Error vs Mesh Size'
set xlabel 'h'
set ylabel 'Error'
set logscale
set key bottom right
p 'space_3_time_1.csv' u 2:3 w lp t 'space 3 time 1',\
  'space_3_time_2.csv' u 2:3 w lp t 'space 3 time 2', \
  'space_3_time_3.csv' u 2:3 w lp t 'space 3 time 3', \
  'space_3_time_4.csv' u 2:3 w lp t 'space 3 time 4', \
