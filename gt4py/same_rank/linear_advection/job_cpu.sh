#!/bin/bash -l

#SBATCH -C gpu -A csstaff -p debug -o  out.out

srun -J job1 python main.py 10 0 1 cpu
srun -J job2 python main.py 10 0 1 cpu
