#!/bin/bash
#SBATCH --account=project_462000559
#SBATCH --partition=dev-g
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=120G
#SBATCH --time=00:05:00

#export EBU_USER_PREFIX=/project/project_462000559/EasyBuild
# ml LUMI/23.09
# module load PyTorch/2.2.0-rocm-5.6.1-python-3.10-asterix-singularity-20240315

module purge
module use /appl/local/csc/modulefiles
module load pytorch



# srun singularity exec $SIF 
srun torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=2 \
  vq_vae_dist.py ../kostis/khi/control/restart.0000100.2024-05-31_12-50-15.vlsv
