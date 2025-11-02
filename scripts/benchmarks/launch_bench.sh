#!/bin/bash

# Usage: ./submit_n_times.sh <N>


if [ $# -ne 2 ]; then
  echo "Usage: $0 <N> <GPUS>"
  exit 1
fi

N=$1
GPUS=$2

for ((i=1; i<=N; i++)); do
  echo "Submitting job $i of $N..."
  sbatch scripts/benchmarks/benchmark_vnccl_${GPUS}.sh
done