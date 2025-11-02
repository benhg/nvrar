#!/usr/bin/env python3
# Copyright 2025 Parallel Software and Systems Group, University of Maryland.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch

import os 
import torch.distributed as dist
from torch.profiler import record_function
from mpi4py import MPI

def init():
    rank = int(os.getenv("SLURM_PROCID", 0))
    world_size = int(os.getenv("SLURM_NTASKS", 1))
    dist.init_process_group(rank=rank, 
                            world_size=world_size,
                            backend="nccl", 
                            init_method="env://")
    torch.cuda.set_device(rank %  torch.cuda.device_count())
    

def time_something(fn, *args, warmup_iters=5, timed_iters=20, prof=None, **kwargs):
    start_event = torch.cuda.Event(enable_timing=True) 
    end_event = torch.cuda.Event(enable_timing=True) 
    # warmup iters
    for i in range(warmup_iters):
        with record_function(f"warmup_{i}"):
            fn(*args, **kwargs)
        #if prof is not None:
        #prof.step()
    torch.cuda.synchronize()
    # Start timing with both CUDA events and MPI_Wtime
    mpi_start_time = MPI.Wtime()
    start_event.record()
    # timed iters
    for i in range(timed_iters):
        with record_function(f"iteration_{i}"):
            fn(*args, **kwargs)
        #if prof is not None:
        #prof.step()
    # End timing with CUDA events and MPI_Wtime
    end_event.record()
    torch.cuda.synchronize()
    mpi_end_time = MPI.Wtime()

    cuda_event_time_ms = start_event.elapsed_time(end_event) / timed_iters
    mpi_Wtime_ms = (mpi_end_time - mpi_start_time) * 1000.0 / timed_iters
    
    return cuda_event_time_ms, mpi_Wtime_ms
