#!/usr/bin/env python3
# Copyright 2025 Parallel Software and Systems Group, University of Maryland.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
import torch.distributed as dist
from . import nvshmem_comm_cuda

class NVRARCommunicator:
    def __init__(self, process_group: torch.distributed.ProcessGroup):
        rank = torch.distributed.get_rank(process_group)
        nranks = torch.distributed.get_world_size(process_group)

        if nranks == 1:
            return

        device = torch.cuda.current_device()

        unique_id = nvshmem_comm_cuda.NVSHMEMCommWrapper.get_unique_id_bytes()
        uid_gpu = unique_id.to("cuda")
        ranks = torch.distributed.get_process_group_ranks(process_group)
        torch.distributed.broadcast(uid_gpu, src=ranks[0], group=process_group)
        torch.distributed.barrier(group=process_group)

        unique_id = uid_gpu.to("cpu")

        self.comm_wrapper = nvshmem_comm_cuda.NVSHMEMCommWrapper(rank, nranks, device, unique_id)
        print(f"NVRARCommunicator created for process group {process_group} with rank {rank} and nranks {nranks}")

        self.comm_wrapper.set_kernel_params(4, 256, 16384)

    @property
    def core(self):
        return self.comm_wrapper


