#!/bin/bash -l

#SBATCH -C mc -A csstaff -o output/sine/space_4_time_4.out -t 24:0:0

rank=3
runge=4

srun python main.py 20 $rank $runge cpu
srun python main.py 40 $rank $runge cpu
srun python main.py 80 $rank $runge cpu
srun python main.py 160 $rank $runge cpu
srun python main.py 320 $rank $runge cpu
srun python main.py 640 $rank $runge cpu
srun python main.py 1280 $rank $runge cpu
srun python main.py 2560 $rank $runge cpu
srun python main.py 5120 $rank $runge cpu
