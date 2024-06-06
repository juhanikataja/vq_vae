#!/bin/bash
#SBATCH --account=project_462000559
#SBATCH --partition=dev-g
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=7
#SBATCH --mem=120G
#SBATCH --time=00:05:00

#export EBU_USER_PREFIX=/project/project_462000559/EasyBuild
ml LUMI/23.09
module load PyTorch/2.2.0-rocm-5.6.1-python-3.10-asterix-singularity-20240315

export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
export MASTER_PORT=24500
export WORLD_SIZE=$SLURM_NPROCS

srun bash -c "RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID singularity exec $SIF python \
  vq_vae_dist.py ../kostis/khi/control/restart.0000100.2024-05-31_12-50-15.vlsv"
