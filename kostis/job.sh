#!/bin/bash
#SBATCH --account=project_462000559
#SBATCH --partition=standard-g
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=8
#SBATCH --mem=480G
#SBATCH --time=1:00:00

export RDZV_HOST=$(hostname)
export RDZV_PORT=9999

# module purge
# module use /appl/local/csc/modulefiles
# module load pytorch


module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.0




srun python3 -m torch.distributed.run --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=8 --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint="$RDZV_HOST:$RDZV_PORT"  vq_vae3_mpi.py restart.0000200.2024-05-31_12-51-47.vlsv
