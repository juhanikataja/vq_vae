#!/bin/bash
#SBATCH --account=project_462000559
#SBATCH --partition=dev-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --time=0:05:00
#SBATCH --gpus-per-node=4

module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.0

srun python3 -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
    vq_vae_dist.py data.vlsv
