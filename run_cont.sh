#!/bin/bash
################################
## SLURM batchjob script for
## Elmer on LUMI
##
## copyleft 2023-06-21
##    CSC-IT Center for Sciencce
##
################################
 
#SBATCH --time=00:15:00
#SBATCH --job-name=vqvae
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=dev-g

####### change to your project #######
#SBATCH --account=project_462000559

####### change to numbers of nodes and MPI tasks ###
####### NB: we provide meshes for 128,256,512 and 1024 partitions #####
#######     do the math by matching the product of next entries   #####
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=250G
#SBATCH --exclusive

################## OpenMP Stuff ##########
## use only if you undersubscribe
## the MPI tasks
##########################################
#SBATCH --cpus-per-task=1
#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
#echo "running OpenMP on $SLURM_CPUS_PER_TASK"
#export KMP_AFFINITY=compact
#export KMP_DETERMINISTIC_REDUCTION=yes

#export CRAY_ACC_NO_ASYNC=1

## These control USM behaviour
#export CRAY_ACC_USE_UNIFIED_MEM=0
#export HSA_XNACK=0
#export CRAY_ACC_DEBUG=0
#export LIBOMPTARGET_KERNEL_TRACE=0

#export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
#export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb=10"
###### enable CSC provided modules #########

#module use /appl/local/csc/modulefiles 
#ml pytorch/2.2



export EBU_USER_PREFIX=/project/project_462000559/EasyBuild
ml LUMI/23.09
module load PyTorch/2.2.0-rocm-5.6.1-python-3.10-asterix-singularity-20240315

#export CXI_FORK_SAFE=1

## Analysator
export PYTHONPATH=$PYTHONPATH:/scratch/project_462000559/kostis/libs/analysator
rocminfo
srun singularity exec --bind $(pwd) $SIF python3 vq_vae2.py restart.0000100.2024-05-31_12-50-15.vlsv
 
