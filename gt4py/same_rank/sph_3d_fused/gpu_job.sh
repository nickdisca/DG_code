#!/bin/bash 

#SBATCH -C gpu -A csstaff -t 24:0:0


#SBATCH -J gpu_nx640
#SBATCH -o output/gpu_nx640.out

srun python main.py 640 1 3 4 gt:gpu
srun python main.py 640 2 3 4 gt:gpu
srun python main.py 640 3 3 4 gt:gpu
srun python main.py 640 4 3 4 gt:gpu
srun python main.py 640 8 3 4 gt:gpu
srun python main.py 640 16 3 4 gt:gpu

# srun python main.py 10 1 3 4 gt:gpu
# srun python main.py 20 1 3 4 gt:gpu
# srun python main.py 40 1 3 4 gt:gpu
# srun python main.py 80 1 3 4 gt:gpu
# srun python main.py 160 1 3 4 gt:gpu
# srun python main.py 320 1 3 4 gt:gpu
# srun python main.py 640 1 3 4 gt:gpu
# srun scalene --profile-all --cli --output cpu.prof main.py
# srun python -m cProfile -o nx300_20_gpu.prof main.py 300 20 3 4 gt:gpu
