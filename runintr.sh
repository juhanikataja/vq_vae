export CRAY_ACC_USE_UNIFIED_MEM=0
export HSA_XNACK=0
alias srme='srun -t 00:05:00 -J asterix_vqvae_test -p dev-g -A project_462000559 -N 1 -n 1 -G 1'
