#!/usr/bin/env python3
# Copyright 2025 Parallel Software and Systems Group, University of Maryland.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
import torch.distributed as dist
import nvshmem.core
from cuda.core.experimental import Device
from . import nvshmem_comm_cuda

class NVRARCommunicator:
    def __init__(self, process_group: torch.distributed.ProcessGroup):
        rank = torch.distributed.get_rank(process_group)
        nranks = torch.distributed.get_world_size(process_group)

        if nranks == 1:
            return

        device = torch.cuda.current_device()
        cuda_dev = Device(device)
        # This should be idempotent
        cuda_dev.set_current()
        stream = torch.cuda.current_stream()

        # Fetch NVSHMEM unique ID via nvshmem4py and broadcast via torch.distributed
        uniqueid = nvshmem.core.get_unique_id(empty=True)
        if rank == 0:
            # Rank 0 gets a real uniqueid
            uniqueid = nvshmem.core.get_unique_id()
            broadcast_objects = [uniqueid]
        else:
            broadcast_objects = [None]

        # We use torch.distributed.broadcast_object_list to send the UID to all ranks
        dist.broadcast_object_list(broadcast_objects, src=0, group=process_group)
        dist.barrier(group=process_group)

        nvshmem.core.init(
            device=cuda_dev,
            uid=broadcast_objects[0],
            rank=rank,
            nranks=nranks,
            initializer_method="uid",
        )
       

        self.comm_wrapper = nvshmem_comm_cuda.NVSHMEMCommWrapper(rank, nranks, device)
        print(f"NVRARCommunicator created for process group {process_group} with rank {rank} and nranks {nranks}")

        self.comm_wrapper.set_kernel_params(4, 256, 16384)

    @property
    def core(self):
        return self.comm_wrapper


