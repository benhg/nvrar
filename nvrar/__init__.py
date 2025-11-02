#!/usr/bin/env python3
# Copyright 2025 Parallel Software and Systems Group, University of Maryland.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

try:
    from . import nvshmem_comm_cuda  # available if NVSHMEM build succeeded
    from .comm import NVRARCommunicator
    NVRAR_AVAILABLE = True
except Exception as e:
    print(f"Failed to import nvshmem_comm_cuda extension: {e}")
    nvshmem_comm_cuda = None  # type: ignore
    NVRAR_AVAILABLE = False

from .config_paths import NVRAR_CACHE_DIR
from .config import resolve_params

