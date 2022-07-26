#!/bin/bash -l

#SBATCH -C mc -A csstaff -p normal -t 24:0:0 -o order_3.out

srun python main.py 20 2 3 cpu
srun python main.py 40 2 3 cpu
srun python main.py 80 2 3 cpu
srun python main.py 160 2 3 cpu
srun python main.py 320 2 3 cpu
srun python main.py 640 2 3 cpu
srun python main.py 1280 2 3 cpu
srun python main.py 2560 2 3 cpu
