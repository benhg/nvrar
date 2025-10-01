try:
    from . import nvshmem_comm_cuda  # available if NVSHMEM build succeeded
    from .comm import NVRARCommunicator
    NVRAR_AVAILABLE = True
except Exception:
    nvshmem_comm_cuda = None  # type: ignore
    NVRAR_AVAILABLE = False


