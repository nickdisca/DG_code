set logscale
set key bottom right
p 'space_1_time_1.csv' u 2:3 w lp t 'space 1 time 1',\
  'space_1_time_2.csv' u 2:3 w lp t 'space 1 time 2',\
  'space_2_time_2.csv' u 2:3 w lp t 'space 2 time 2', \
  'space_3_time_1.csv' u 2:3 w lp t 'space 3 time 1', \
  'space_3_time_2.csv' u 2:3 w lp t 'space 3 time 2', \
  'space_3_time_3.csv' u 2:3 w lp t 'space 3 time 3', \
