#!/bin/bash -l

#SBATCH -C mc -A csstaff -o output/sine/space_3_time_2.out -t 24:0:0

rank=2
runge=2

srun python main.py 20 $rank $runge cpu
srun python main.py 40 $rank $runge cpu
srun python main.py 80 $rank $runge cpu
srun python main.py 160 $rank $runge cpu