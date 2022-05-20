#!/bin/bash -l

#SBATCH -C gpu -A csstaff -o gpu.out -t 5:0:0

srun python main.py 1280 0 1 cuda
srun python main.py 2560 0 1 cuda
