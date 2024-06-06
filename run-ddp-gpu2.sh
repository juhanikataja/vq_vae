#!/bin/bash
#SBATCH --account=project_462000559
#SBATCH --partition=dev-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=2
#SBATCH --gpus-per-task=2
#SBATCH --mem=120G
#SBATCH --time=0:05:00

export EBU_USER_PREFIX=/project/project_462000559/EasyBuild
ml LUMI/23.09
module load PyTorch/2.2.0-rocm-5.6.1-python-3.10-asterix-singularity-20240315

# module purge
# module use /appl/local/csc/modulefiles
# module load pytorch

export NCCL_SOCKET_IFNAME=hsn

# Old way with torch.distributed.run
# srun singularity exec --bind $(pwd) $CONTAINER \
#   python3 -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \
#   vq_vae2.py ../kostis/khi/control/restart.0000100.2024-05-31_12-50-15.vlsv


# srun python3 -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \
#   vq_vae2.py ../kostis/khi/control/restart.0000100.2024-05-31_12-50-15.vlsv


export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
export MASTER_PORT=24500
export WORLD_SIZE=$SLURM_NPROCS

export SINGULARITYENV_NCCL_DEBUG=INFO
export SINGULARITYENV_NCCL_DEBUG_SUBSYS=INIT,COLL

# Container+torch.distributed.run
srun bash -c "RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID singularity exec --bind $(pwd) $SIF \
  python3 -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 \
           vq_vae_dist.py ../kostis/khi/control/restart.0000100.2024-05-31_12-50-15.vlsv"


# New way with torchrun
# srun torchrun --standalone --nnodes=1 --nproc_per_node=4 mnist_ddp.py --epochs=100 
