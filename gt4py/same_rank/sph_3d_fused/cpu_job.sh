#!/bin/bash 

#SBATCH -C mc -o output/cpu_ifirst.out -A csstaff -t 24:0:0 

srun python main.py 100 40 3 4 gt:cpu_ifirst
# srun scalene --profile-all --cli --output cpu.prof main.py
# srun python -m cProfile -o cprofile_dace_cpu.prof main.py
