#!/bin/bash
#SBATCH --gpus-per-node=4
#SBATCH -A m5083_g
#SBATCH --nodes=2
#SBATCH --qos regular
#SBATCH --time 00:19:00
#SBATCH --ntasks-per-node=4
#SBATCH --constraint="gpu"

module load PrgEnv-gnu cudatoolkit/12.9.lua nvshmem/ python/3.12
. $SCRATCH/venvs/mlsys-env/bin/activate

NNODES=$SLURM_JOB_NUM_NODES
GPUS=$(( NNODES * 4 ))
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29502
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NET_GDR_LEVEL=PHB
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=3,2,1,0
export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET="AWS Libfabric"
export FI_CXI_RX_MATCH_MODE=hybrid      # try software if queues exhaust

export NVSHMEM_REMOTE_TRANSPORT=libfabric
export NVSHMEM_DISABLE_CUDA_VMM=1
export FI_PROVIDER=cxi


export FI_CXI_RDZV_THRESHOLD=0
#export FI_CXI_RDZV_THRESHOLD=262144
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_OFLOW_BUF_SIZE=1073741824
export FI_CXI_OFLOW_BUF_COUNT=10
export MPICH_GPU_SUPPORT_ENABLED=1

export HF_HOME="$SCRATCH/hf_cache"
export TRANSFORMERS_HOME="$SCRATCH/hf_cache"
export HF_DATASETS_CACHE="$SCRATCH/hf_cache"
export YALIS_CACHE="/pscratch/sd/p/prajwal/SpecDec/yalis/yalis/external"

CUDA_VISIBLE_DEVICES=3,2,1,0 

srun -N $NNODES -n $GPUS -c 24 --gpus-per-node 4 --cpu-bind=cores --exclusive ./scripts/get_rank.sh python -u benchmarks/benchmark_compare_allreduce.py --sizes 64KiB,128KiB,256KiB,512KiB,1MiB,2MiB --dtype bfloat16 --iterations 1000 --warmup 200 --graphs --graph-inner-iters 100 | tee nvrar_v_nccl_${GPUS}_${SLURM_JOB_ID}_graphs.out 2>&1
srun -N $NNODES -n $GPUS -c 24 --gpus-per-node 4 --cpu-bind=cores --exclusive ./scripts/get_rank.sh python -u benchmarks/benchmark_compare_allreduce.py --sizes 64KiB,128KiB,256KiB,512KiB,1MiB,2MiB --dtype bfloat16 --iterations 1000 --warmup 200 | tee nvrar_v_nccl_${GPUS}_${SLURM_JOB_ID}.out 2>&1

