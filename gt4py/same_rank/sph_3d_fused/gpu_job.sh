#!/bin/bash 

#SBATCH -C gpu -o output/profile_dace_gpu.out -A csstaff -t 24:0:0 

# srun python main.py 100 40 3 4 dace:gpu
# srun scalene --profile-all --cli --output cpu.prof main.py
srun python -m cProfile -o dace_gpu_fused.prof main.py
