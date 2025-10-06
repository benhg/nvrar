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

