#!/usr/bin/env python3
# Copyright 2025 Parallel Software and Systems Group, University of Maryland.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Simple test to check nvshmem4py PE querying and symmetric tensor allocation/free.
Run with: torchrun --nproc_per_node=4 tests/test_rank_worldsize_uid.py
"""

import sys
import os
import torch
import numpy as np
from nvshmem import core as nvshmem
from cuda.core.experimental import Device


def test_nvshmem4py_basics():
    """Test nvshmem4py my_pe/n_pes and symmetric tensor alloc/free."""
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    local_rank = rank % 4

    print(f"\n=== Testing nvshmem4py (Rank {rank}/{world_size-1}) ===")

    try:
        # Initialize device
        torch.cuda.set_device(torch.device(f"cuda:{local_rank}"))
        cuda_dev = Device(local_rank)
        cuda_dev.set_current()

        # Initialize nvshmem4py via UID method
        uniqueid = nvshmem.get_unique_id(empty=True)
        if rank == 0:
            uniqueid = nvshmem.get_unique_id()
            obj = [uniqueid]
        else:
            obj = [None]
        torch.distributed.broadcast_object_list(obj, src=0)
        torch.distributed.barrier()
        nvshmem.init(device=cuda_dev, uid=obj[0], rank=rank, nranks=world_size, initializer_method="uid")

        # Query PEs from nvshmem4py
        my_pe = nvshmem.my_pe()
        n_pes = nvshmem.n_pes()
        print(f"✓ nvshmem.my_pe()={my_pe}, nvshmem.n_pes()={n_pes}")

        torch.distributed.barrier()

        # Basic sanity: world sizes should match
        if n_pes != world_size:
            print(f"✗ nvshmem.n_pes() != torch.distributed world_size: {n_pes} vs {world_size}")
            return False
        else:
            print("✓ nvshmem.n_pes() matches torch.distributed world_size")

        # Allocate symmetric tensor with nvshmem and free it
        t = nvshmem.tensor((4096,), dtype=torch.float16)
        t.fill_(1)
        torch.cuda.synchronize()

        # No collective here; just ensure allocation and free work
        nvshmem.free_tensor(t)
        torch.cuda.synchronize()
        print(f"✓ nvshmem.tensor/free_tensor succeeded on rank {rank}")
        return True

    except Exception as e:
        print(f"✗ Error during testing on rank {rank}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Initialize MPI
    torch.distributed.init_process_group(backend="nccl")
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    local_rank = rank % 4
    torch.cuda.set_device(torch.device(f"cuda:{local_rank}"))
    
    if rank == 0:
        print("Testing nvshmem4py with Torch Process groups")
        print(f"Running with {world_size} processes")
        print("=" * 60)
    
    # Synchronize all processes
    torch.distributed.barrier()
    
    success = test_nvshmem4py_basics()

    success_t = torch.tensor(int(success), device=f"cuda:{local_rank}", dtype=torch.int32)
    torch.distributed.all_reduce(success_t, op=torch.distributed.ReduceOp.MIN)  # MIN==1 only if everyone had 1
    if success_t.item() == 1:
        print("\n🎉 All tests passed successfully on all ranks!")
    else:
        print("\n❌ Some tests failed!")    

    # Non-root processes wait for root to exit
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


