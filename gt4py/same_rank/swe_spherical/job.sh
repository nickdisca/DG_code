#!/bin/bash 

#SBATCH -C gpu -o output/gpu.out -A csstaff -t 24:0:0 

srun python main.py 20 3 4 cpu
