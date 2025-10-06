## 🚀 NVRAR: NVSHMEM-based Recursive All-Reduce for Low-Latency LLM Inference

NVRAR is a PyTorch extension that implements a low-latency, GPU-resident
all-reduce using NVSHMEM. It targets the small-message inter-node regime, aiming
to minimize inference latency in the *Decode* phase of LLM inference when using
tensor parallelism. The project builds a Python module (`nvrar`) with CUDA/C++
kernels and a thin CMake-based build powered by scikit-build-core.

### ✨ Features
- **NVSHMEM-backed** device-initiated collectives
- **Low-latency hierarchical all-reduce** for small messages based on recursive reduce across nodes
- **Preallocated-tensor API** for avoiding per-call allocations
- **PyTorch integration** via `torch.distributed` for UID exchange and synchronization
- Tuning script to discover best kernel launch parameters per message size and dtype


## 🧰 Prerequisites
You will need the following on your system:
- **CUDA** toolkit and a compatible NVIDIA GPU
- **PyTorch** with CUDA support (`pip install torch` appropriate to your CUDA)
- **CMake >= 3.24**, **Ninja**, **pybind11**, and **scikit-build-core** (handled by build)
- **MPI** (headers and libs; used during build/link)
- **NVSHMEM >= 3.2.5** installed; set one of:
  - export `NVSHMEM_HOME=/path/to/nvshmem`
  - or configure with `-DNVSHMEM_ROOT=/path/to/nvshmem`


Note: The CMake build detects CUDA architecture but in case of failure, use `-DCMAKE_CUDA_ARCHITECTURES=<arch no. like 80/90>` to override.


## 📦 Installation
You can install from the repo root using pip. Ensure `NVSHMEM_HOME` (or `NVSHMEM_ROOT`) is set and that your PyTorch matches your CUDA.

Minimal install:

```bash
export NVSHMEM_HOME=/opt/nvshmem           # adjust path to NVSHMEM installation
PIP_NO_BUILD_ISOLATION=1 pip install -v -e .
```

Specify CUDA archs or extra CMake options via `CMAKE_ARGS`:

```bash

export NVSHMEM_HOME=/opt/nvshmem
export CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=80;86;90 -DNVSHMEM_ROOT=$NVSHMEM_HOME"
# Optional
export CC=<Path to gcc>
export CXX=<Path to g++>

PIP_NO_BUILD_ISOLATION=1 pip install -v -e .
```

If building wheels in a clean environment, scikit-build-core will fetch CMake/Ninja as needed (per `pyproject.toml`).


## ⚡ Quick Start
After install, verify the extension loads and initialize a communicator on a `torch.distributed` process group.

```python
import torch
import torch.distributed as dist
from nvrar import NVRAR_AVAILABLE

print("NVRAR available:", NVRAR_AVAILABLE)
```

For multi-rank runs, you can use `torchrun` to initialize ranks and then use the provided tuning script.


## 🎛️ Tuning Kernel Parameters
The tuner explores grid configurations for `(num_blocks, threads_per_block, chunk_bytes)` and validates correctness and latency across ranks.

Script: `tuning/benchmark_tune_allreduce_preallocated.py`

Example (single size in bytes):

```bash
torchrun --nproc_per_node=4 tuning/benchmark_tune_allreduce_preallocated.py \
  --size 1048576 --dtype float32 \
  --num-blocks 4,8,16,32 --threads-per-block 128,256,512 \
  --chunk-bytes 16384,32768,65536,131072,262144 \
  --iterations 50 --warmup 10 --topk 10
```

Multiple sizes with human-readable tokens:

```bash
torchrun --nproc_per_node=4 tuning/benchmark_tune_allreduce_preallocated.py \
  --sizes 64KiB,1MiB,8MiB --dtype float32 \
  --output tune_results.json
```


Notes:
- The tuner uses NCCL via `torch.distributed` for UID broadcast and reductions.
- On success, best-per-size configs can be written into the cache directory (see below) and auto-resolved by the library.


## 🧩 Using the Library
High-level flow:
1. Initialize `torch.distributed` (e.g., `torchrun --nproc_per_node=N ...`).
2. Create or reuse a `ProcessGroup`.
3. Construct the NVSHMEM communicator wrapper and perform operations.

The package exports helpers in `nvrar`:
- `NVRAR_AVAILABLE`: import-time flag indicating if the native extension loaded
- `NVRAR_CACHE_DIR`: path used to store tuned parameter JSON files
- `resolve_params(num_gpus, dtype)`: selects tuned parameters or defaults


## 🗄️ Cache and Configuration
The library uses a per-user cache directory for tuned parameters (see `nvrar/config_paths.py`).

- Cache dir: `NVRAR_CACHE_DIR` env or OS default (e.g., `~/.cache/nvrar`)
- Env override (for direct JSON path): `NVSHMEM_ALLREDUCE_CONFIG`
- Default tuned filename key: `tuning_{world_size}gpu_{dtype}.json`

#### 🌐 Environment Variables
- `NVRAR_CACHE_DIR`: overrides cache directory for tuned files.
- `NVSHMEM_ALLREDUCE_CONFIG`: path to a JSON file with per-size best parameters.


--- 

## 📄 License
Apache-2.0 WITH LLVM-exception. See `LICENSE`.


