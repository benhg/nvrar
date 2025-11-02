#!/bin/bash
# select_gpu_device wrapper script
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NET_GDR_LEVEL=PHB
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=3,2,1,0
export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=hsn
export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=hsn0
export NCCL_NET="AWS Libfabric"
export NCCL_GRAPH_MIXING_SUPPORT=0 # This is very important for performance

export NVSHMEM_REMOTE_TRANSPORT=libfabric
export NVSHMEM_DISABLE_CUDA_VMM=1
#export FI_CXI_OPTIMIZED_MRS=false
export FI_PROVIDER=cxi
#export FI_HMEM_CUDA_USE_GDRCOPY=1
#export FI_CXI_LLRING_MODE=idle

# Setting to 0 is important for torch dist to work
export FI_CXI_RDZV_THRESHOLD=0
#export FI_CXI_RDZV_THRESHOLD=262144
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_OFLOW_BUF_SIZE=1073741824
export FI_CXI_OFLOW_BUF_COUNT=10
export MPICH_GPU_SUPPORT_ENABLED=0

#export FI_CXI_RDZV_THRESHOLD=0
#export FI_CXI_RDZV_GET_MIN=0
#export FI_CXI_RDZV_EAGER_SIZE=0

export JOBID=${SLURM_JOB_ID}
export RANK=${SLURM_PROCID}
export WORLD_SIZE=${SLURM_NTASKS}
export LOCAL_RANK=${SLURM_LOCALID}
export TORCHINDUCTOR_CACHE_DIR="/dev/shm/$USER/.cache/torchinductor/torchinductor_${RANK}"
export TRITON_HOME="/dev/shm/$USER/.cache/triton/triton_${RANK}"
export TRITON_CACHE_DIR="/dev/shm/$USER/.cache/triton/triton_${RANK}"

#export NCCL_DEBUG=INFO
#export NVSHMEM_BOOTSTRAP=plugin
#export NCCL_DEBUG_SUBSYS=COLL,TUNING
#export NVSHMEM_DEBUG=INFO
export NVSHMEM_REMOTE_TRANSPORT=libfabric

#exec nsys profile -o yalistrace -t cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop --cuda-graph-trace=node $*
exec $*
