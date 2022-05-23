#!/bin/bash -l

#SBATCH -C gpu -A csstaff -o order_3_gpu.out -t 24:0:0

srun python main.py 1280 2 3 cuda
srun python main.py 2560 2 3 cuda
srun python main.py 5120 2 3 cuda
srun python main.py 10240 2 3 cuda
