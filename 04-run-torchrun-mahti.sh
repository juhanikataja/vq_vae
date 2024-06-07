#!/bin/bash
#SBATCH --account=project_2002078
#SBATCH --partition=gputest
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=0:05:00
#SBATCH --gres=gpu:a100:4

module purge
module load pytorch

srun python3 -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
    vq_vae_dist.py data.vlsv
