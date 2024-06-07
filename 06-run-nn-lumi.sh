#!/bin/bash
#SBATCH --account=project_462000559
#SBATCH --partition=standard-g
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=2
#SBATCH --cpus-per-task=56
#SBATCH --time=0:05:00
#SBATCH --gpus-per-node=8
#SBATCH --mem=480G

export RDZV_HOST=$(hostname)
export RDZV_PORT=29400

module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.0

srun python3 -m torch.distributed.run \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=8 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$RDZV_HOST:$RDZV_PORT" \
    vq_vae_dist.py data.vlsv
