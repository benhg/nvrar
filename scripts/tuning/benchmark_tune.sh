#!/bin/bash
#SBATCH --gpus-per-node=4
#SBATCH -A m5083_g
#SBATCH --nodes=8
#SBATCH --qos regular
#SBATCH --time 00:19:00
#SBATCH --ntasks-per-node=4
#SBATCH --constraint="gpu"


module load PrgEnv-gnu/ cudatoolkit/12.9.lua nvshmem/
module load conda/Miniforge3-24.11.3-0
conda activate /pscratch/sd/p/prajwal/conda/pytorch-nightly

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
#export NCCL_CUMEM_ENABLE=0
export FI_CXI_RX_MATCH_MODE=hybrid      # try software if queues exhaust

export NVSHMEM_REMOTE_TRANSPORT=libfabric
export NVSHMEM_DISABLE_CUDA_VMM=1
#export FI_CXI_OPTIMIZED_MRS=false
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
#export TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1

CUDA_VISIBLE_DEVICES=3,2,1,0 



srun --qos regular --mpi=cray_shasta -N 8 -n 32 -c 24 --cpu-bind=cores ./get_rank.sh python -u tuning/tune_allreduce_preallocated.py --sizes 8KiB,64KiB,128KiB,256KiB,512KiB,1MiB --dtype bfloat16  --num-blocks 1,2,4,8,16,32 --threads-per-block 512 --chunk-bytes 4096,8192,16384,32768,65536,131072,262144,524288 --iterations 50 --warmup 10 --topk 10 --best-output tuning_32gpu.json
