#!/bin/bash 

#SBATCH -C mc -A csstaff -t 24:0:0

#SBATCH -J cpu_nx640
#SBATCH -o output/cpu_nx640.out

srun python main.py 640 1 3 4 gt:cpu_ifirst
srun python main.py 640 2 3 4 gt:cpu_ifirst
srun python main.py 640 3 3 4 gt:cpu_ifirst
srun python main.py 640 4 3 4 gt:cpu_ifirst
srun python main.py 640 8 3 4 gt:cpu_ifirst
srun python main.py 640 16 3 4 gt:cpu_ifirst


# srun python main.py 640 20 3 4 gt:cpu_ifirst
# srun python main.py 300 20 3 4 gt:cpu_ifirst
# srun scalene --profile-all --cli --output cpu.prof main.py
# srun python -m cProfile -o nx300_nz20_cpu.prof main.py 300 20 3 4 gt:cpu_ifirst
