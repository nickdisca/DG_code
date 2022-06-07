#!/bin/bash 

#SBATCH -C gpu -A csstaff -t 24:0:0 -o output/gpu.out

srun python main.py 100 20 3 4 cuda
