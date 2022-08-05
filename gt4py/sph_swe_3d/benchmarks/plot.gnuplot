set terminal qt size 700,500 enhanced font ', 16'
unset grid
set title 'Execution time vs Problem size'
set xlabel 'Gridpoints'
set ylabel 'Execution Time [s]'
#set xtics 0.5
set key top left Left reverse box

set logscale
set autoscale fix
set key spacing 1.3

f(x) = 0.0005*x**2
p 'benchmark_comparison_nz1.csv' u 1:2 w lp t 'gt:cpu\_ifirst',\
'' u 1:3 w lp t 'gt:cpu\_kfirst', \
'' u 1:4 w lp t 'gt:gpu', \
'' u 1:5 w lp t 'cuda', \
'' u 1:6 w lp t 'dace:cpu', \
'' u 1:7 w lp lw 2 t 'matlab', \
f(x) dashtype 3 lw 3 lc 'gray' t 'O(N^2)'

#f(x) = x**1
#p 'benchmark_comparison_nx300.csv' u 1:2 w lp t 'gt:cpu\_ifirst',\
#'' u 1:3 w lp t 'gt:gpu', \
#'' u 1:4 w lp t 'cuda', \
#f(x) dashtype 3 lw 3 lc 'gray' t 'O(N)'

#f(x) = 0.000002*x**2
#p 'benchmark_scalar_nz1.csv' u 1:2 w lp t 'gt:cpu\_ifirst',\
#'' u 1:3 w lp t 'gt:gpu', \
#f(x) dashtype 3 lw 3 lc 'gray' t 'O(N^2)'

