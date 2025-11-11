#!/usr/bin/env python3
# Copyright 2025 Parallel Software and Systems Group, University of Maryland.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Benchmark and compare NVRAR (NVSHMEM) all-reduce vs torch.distributed all_reduce.

Usage examples:
  # Single size in bytes
  torchrun --nproc_per_node=4 benchmarks/benchmark_compare_allreduce.py \
    --size 1048576 --dtype float32 --iterations 100 --warmup 10

  # Multiple sizes (bytes with suffixes are supported: KiB, MiB, GiB)
  torchrun --nproc_per_node=4 benchmarks/benchmark_compare_allreduce.py \
    --sizes 64KiB,1MiB,8MiB --dtype float32 --iterations 50 --warmup 10

  # Enable CUDA Graphs capture for both paths
  torchrun --nproc_per_node=4 benchmarks/benchmark_compare_allreduce.py \
    --sizes 64KiB,1MiB --dtype float32 --iterations 100 --warmup 10 --graphs

The script reports average latency per iteration (ms) across ranks, using the
max across ranks as the collective latency for each configuration.
"""

import argparse
import os
import sys
from typing import List

import torch
import torch.distributed as dist
from nvshmem import core as nvshmem
from cuda.core.experimental import Device

try:
    from nvrar import nvshmem_comm_cuda, resolve_params
except Exception as e:  # pragma: no cover
    print(f"Failed to import nvrar extension or helpers: {e}")
    nvshmem_comm_cuda = None  # type: ignore
    resolve_params = None  # type: ignore

# BEGIN INSERT: import time_something utility
try:
    from benchmarks.utils import time_something  # type: ignore
except Exception:
    import os as _os, sys as _sys
    _sys.path.append(_os.path.dirname(__file__))
    from utils import time_something  # type: ignore
# END INSERT


DTYPE_MAP = {
    "int32": torch.int32,
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def parse_size_to_bytes(token: str) -> int:
    s = token.strip().replace("_", "").lower()
    if not s:
        raise ValueError("empty size token")
    if s.isdigit():
        return int(s)
    i = 0
    while i < len(s) and (s[i].isdigit() or s[i] == "."):
        i += 1
    num_str = s[:i]
    unit = s[i:]
    value = float(num_str)
    if unit in ("k", "kb"):
        return int(value * (1000 ** 1))
    if unit in ("m", "mb"):
        return int(value * (1000 ** 2))
    if unit in ("g", "gb"):
        return int(value * (1000 ** 3))
    if unit in ("kib", "ki"):
        return int(value * (1024 ** 1))
    if unit in ("mib", "mi"):
        return int(value * (1024 ** 2))
    if unit in ("gib", "gi"):
        return int(value * (1024 ** 3))
    if unit in ("b",):
        return int(value)
    try:
        return int(float(s))
    except Exception as ex:
        raise ValueError(f"Unrecognized size token: {token}") from ex


def parse_size_list(csv: str) -> List[str]:
    return [x for x in (t.strip() for t in csv.split(",")) if x]


def detect_local_device(rank: int) -> int:
    val = os.environ.get("LOCAL_RANK")
    if val is not None:
        try:
            return int(val)
        except ValueError:
            pass
    device_count = torch.cuda.device_count()
    if device_count == 0:
        raise RuntimeError("No CUDA devices available")
    return rank % device_count


def benchmark_torch_allreduce(
    tensor: torch.Tensor,
    iterations: int,
    warmup: int,
    stream: torch.cuda.Stream,
    process_group=None,
) -> tuple[float, float]:
    """Return (avg_ms_event, avg_ms_mpi) per-iter using benchmarks.utils.time_something."""
    # Pre-fill so fill_ cost is excluded
    with torch.cuda.stream(stream):
        tensor.fill_(0.01)
    torch.cuda.synchronize(device=tensor.device)
    dist.barrier(group=process_group)

    def do_one_iter():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=process_group)

    # Run timing on the requested stream
    with torch.cuda.stream(stream):
        dist.barrier(group=process_group)
        evt_ms, mpi_ms = time_something(do_one_iter, warmup_iters=warmup, timed_iters=iterations)

    # Reduce to global max across ranks
    t_evt = torch.tensor([evt_ms], device=tensor.device, dtype=torch.float64)
    dist.all_reduce(t_evt, op=dist.ReduceOp.MAX, group=process_group)
    t_mpi = torch.tensor([mpi_ms], device=tensor.device, dtype=torch.float64)
    dist.all_reduce(t_mpi, op=dist.ReduceOp.MAX, group=process_group)
    return float(t_evt.item()), float(t_mpi.item())


def benchmark_torch_allreduce_graph(
    tensor: torch.Tensor,
    iterations: int,
    warmup: int,
    stream: torch.cuda.Stream,
    process_group=None,
    inner_iters: int = 1,
    use_matmul: bool = False,
) -> tuple[float, float]:
    """Graph-captured torch.distributed all_reduce with optional inner matmul.

    Returns (avg_ms_event, avg_ms_mpi) per logical iteration.
    inner_iters controls how many ops are captured inside a single graph replay.
    Set use_matmul to True to include matmul and feed-forward between inner iters.
    """
    tensor.fill_(0.01)
    dist.barrier(group=process_group)

    # Prime outside capture: run the op warmup times
    with torch.cuda.stream(stream):
        for _ in range(max(0, warmup)):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=process_group)
    torch.cuda.synchronize(device=tensor.device)
    dist.barrier(group=process_group)

    # Prepare optional matmul views/buffers (float dtypes only)
    K = 512
    rows = tensor.numel() // K
    use_matmul = bool(use_matmul) and (rows > 0) and tensor.dtype.is_floating_point
    if use_matmul:
        x_view = tensor.view(rows, K)
        W = torch.randn(K, K, device=tensor.device, dtype=tensor.dtype)
        out_view = torch.empty(rows, K, device=tensor.device, dtype=tensor.dtype)
        # Warm up matmul outside capture to avoid cuBLAS init during capture
        with torch.cuda.stream(stream):
            for _ in range(max(0, warmup)):
                torch.mm(x_view, W, out=out_view)
            x_view.copy_(out_view)
        torch.cuda.synchronize(device=tensor.device)
    else:
        x_view = None
        W = None
        out_view = None

    dist.barrier(group=process_group)

    g = torch.cuda.CUDAGraph()
    # Capture on the same stream used for ops
    with torch.cuda.graph(g, stream=stream):
        for _ in range(max(1, inner_iters)):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=process_group)
            if use_matmul:
                # y = x @ W; write back into x_view for next iteration
                torch.mm(x_view, W, out=out_view)
                x_view.copy_(out_view)

    # Time graph replays via utility
    def replay_once():
        g.replay()

    # Warmup and time using the shared timing utility
    with torch.cuda.stream(stream):
        dist.barrier(group=process_group)
        evt_ms_replay, mpi_ms_replay = time_something(replay_once, warmup_iters=warmup, timed_iters=iterations)
    torch.cuda.synchronize(device=tensor.device)

    # Convert to per-iteration metrics (not per-replay)
    denom = float(max(1, inner_iters))
    evt_ms = evt_ms_replay / denom
    mpi_ms = mpi_ms_replay / denom

    # Reduce to global max across ranks
    t_evt = torch.tensor([evt_ms], device=tensor.device, dtype=torch.float64)
    dist.all_reduce(t_evt, op=dist.ReduceOp.MAX, group=process_group)
    t_mpi = torch.tensor([mpi_ms], device=tensor.device, dtype=torch.float64)
    dist.all_reduce(t_mpi, op=dist.ReduceOp.MAX, group=process_group)
    return float(t_evt.item()), float(t_mpi.item())


def benchmark_nvrar_allreduce(
    comm_wrapper,
    tensor: torch.Tensor,
    tensor_id: int,
    iterations: int,
    warmup: int,
    stream: torch.cuda.Stream,
    algorithm: str,
    stream_ptr,
    process_group=None,
) -> tuple[float, float]:
    # Pre-fill so fill_ cost is excluded
    with torch.cuda.stream(stream):
        tensor.fill_(0.01)
    torch.cuda.synchronize(device=tensor.device)
    dist.barrier(group=process_group)

    def do_one_iter():
        comm_wrapper.allreduce_preallocated(tensor, tensor_id, stream_ptr, algorithm)

    with torch.cuda.stream(stream):
        dist.barrier(group=process_group)
        evt_ms, mpi_ms = time_something(do_one_iter, warmup_iters=warmup, timed_iters=iterations)

    t_evt = torch.tensor([evt_ms], device=tensor.device, dtype=torch.float64)
    dist.all_reduce(t_evt, op=dist.ReduceOp.MAX, group=process_group)
    t_mpi = torch.tensor([mpi_ms], device=tensor.device, dtype=torch.float64)
    dist.all_reduce(t_mpi, op=dist.ReduceOp.MAX, group=process_group)
    return float(t_evt.item()), float(t_mpi.item())


def benchmark_nvrar_allreduce_graph(
    comm_wrapper,
    tensor: torch.Tensor,
    tensor_id: int,
    iterations: int,
    warmup: int,
    stream: torch.cuda.Stream,
    algorithm: str,
    stream_ptr,
    process_group=None,
    inner_iters: int = 1,
    use_matmul: bool = False,
) -> tuple[float, float]:
    tensor.fill_(0.01)
    dist.barrier(group=process_group)

    # Prime outside capture: run the op warmup times
    with torch.cuda.stream(stream):
        for _ in range(max(0, warmup)):
            comm_wrapper.allreduce_preallocated(tensor, tensor_id, stream_ptr, algorithm)
    torch.cuda.synchronize(device=tensor.device)
    dist.barrier(group=process_group)

    # Prepare optional matmul views/buffers (float dtypes only)
    K = 512
    rows = tensor.numel() // K
    use_matmul = bool(use_matmul) and (rows > 0) and tensor.dtype.is_floating_point
    if use_matmul:
        x_view = tensor.view(rows, K)
        W = torch.randn(K, K, device=tensor.device, dtype=tensor.dtype)
        out_view = torch.empty(rows, K, device=tensor.device, dtype=tensor.dtype)
        # Warm up matmul outside capture to avoid cuBLAS init during capture
        with torch.cuda.stream(stream):
            for _ in range(max(0, warmup)):
                torch.mm(x_view, W, out=out_view)
            x_view.copy_(out_view)
        torch.cuda.synchronize(device=tensor.device)
    else:
        x_view = None
        W = None
        out_view = None

    dist.barrier(group=process_group)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=stream):
        for _ in range(max(1, inner_iters)):
            comm_wrapper.allreduce_preallocated(tensor, tensor_id, stream_ptr, algorithm)
            if use_matmul:
                torch.mm(x_view, W, out=out_view)
                x_view.copy_(out_view)

    def replay_once():
        g.replay()

    with torch.cuda.stream(stream):
        dist.barrier(group=process_group)
        evt_ms_replay, mpi_ms_replay = time_something(replay_once, warmup_iters=warmup, timed_iters=iterations)
    torch.cuda.synchronize(device=tensor.device)

    denom = float(max(1, inner_iters))
    evt_ms = evt_ms_replay / denom
    mpi_ms = mpi_ms_replay / denom

    t_evt = torch.tensor([evt_ms], device=tensor.device, dtype=torch.float64)
    dist.all_reduce(t_evt, op=dist.ReduceOp.MAX, group=process_group)
    t_mpi = torch.tensor([mpi_ms], device=tensor.device, dtype=torch.float64)
    dist.all_reduce(t_mpi, op=dist.ReduceOp.MAX, group=process_group)
    return float(t_evt.item()), float(t_mpi.item())


def main():
    parser = argparse.ArgumentParser(description="Compare NVRAR vs torch.distributed all_reduce (sizes in BYTES)")
    parser.add_argument("--size", type=int, required=False, help="Single message size in bytes")
    parser.add_argument("--sizes", type=str, default=None, help="CSV of sizes in bytes (supports 64KiB, 1MiB, 2GB)")
    parser.add_argument("--dtype", choices=list(DTYPE_MAP.keys()), default="float32", help="Tensor dtype")
    parser.add_argument("--iterations", type=int, default=100, help="Benchmark iterations per method")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations per method")
    parser.add_argument("--quiet", action="store_true", help="Reduce non-rank0 logging")
    parser.add_argument("--graphs", action="store_true", help="Use CUDA Graphs capture for both methods")
    parser.add_argument("--graph-inner-iters", type=int, default=1, help="Number of iterations to capture inside a single CUDA graph replay")
    parser.add_argument("--graph-use-matmul", action="store_true", help="Enable matmul on all-reduce output inside CUDA graphs and feed-forward to next iter")

    args = parser.parse_args()

    if not dist.is_available():
        print("torch.distributed not available")
        sys.exit(1)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method=os.environ.get("DIST_INIT_METHOD", "env://"))

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    local_device_idx = detect_local_device(rank)
    device = torch.device(f"cuda:{local_device_idx}")
    torch.cuda.set_device(local_device_idx)

    if rank == 0 and not args.quiet:
        print("NVRAR vs torch.distributed all_reduce benchmark")
        print(f"World size: {world_size}")

    if nvshmem_comm_cuda is None:
        if rank == 0:
            print("NVRAR extension not available; only torch baseline will run.")

    dtype = DTYPE_MAP[args.dtype]
    dtype_size = torch.tensor([], dtype=dtype).element_size()

    sizes_elements: List[int] = []
    if args.sizes is not None and args.sizes.strip():
        tokens = parse_size_list(args.sizes)
        bytes_list = [parse_size_to_bytes(t) for t in tokens]
        for b in bytes_list:
            if b % dtype_size != 0 and rank == 0 and not args.quiet:
                print(f"Warning: size {b} bytes not divisible by dtype size {dtype_size}. Rounding down.")
            sizes_elements.append(b // dtype_size)
    elif args.size is not None:
        if args.size % dtype_size != 0 and rank == 0 and not args.quiet:
            print(f"Warning: --size {args.size} bytes is not divisible by dtype size {dtype_size}. Rounding down.")
        sizes_elements = [args.size // dtype_size]
    else:
        if rank == 0:
            print("Must provide either --size or --sizes")
        sys.exit(1)

    sizes_elements = [n for n in sizes_elements if n > 0]
    if not sizes_elements:
        if rank == 0:
            print("No valid sizes to benchmark.")
        sys.exit(1)

    # Initialize NVSHMEM communicator if available
    comm_wrapper = None
    if nvshmem_comm_cuda is not None:
        # Set device current
        cuda_dev = Device(local_device_idx)
        cuda_dev.set_current()
        # Rank 0 obtains UID; broadcast via object list
        uniqueid = nvshmem.get_unique_id(empty=True)
        if rank == 0:
            uniqueid = nvshmem.get_unique_id()
            obj = [uniqueid]
        else:
            obj = [None]
        dist.broadcast_object_list(obj, src=0)
        dist.barrier()
        # Initialize nvshmem4py
        nvshmem.init(device=cuda_dev, uid=obj[0], rank=rank, nranks=world_size, initializer_method="uid")
        # Construct wrapper without UID (nvshmem already initialized)
        comm_wrapper = nvshmem_comm_cuda.NVSHMEMCommWrapper(rank, world_size, local_device_idx)

    # Use default stream
    stream = torch.cuda.Stream(device=local_device_idx)
    stream_ptr = stream.cuda_stream

    # Use tuned params if available
    params_resolver = None
    if resolve_params is not None:
        params_resolver = resolve_params(world_size, args.dtype)

    for idx, num_elems in enumerate(sizes_elements):
        if rank == 0 and not args.quiet:
            total = len(sizes_elements)
            size_mib = (num_elems * dtype_size) / (1024 ** 2)
            print(f"\n=== Size {idx+1}/{total}: {num_elems} elements ({size_mib:.2f} MiB), dtype={args.dtype} ===")

        # Prepare baseline tensor
        tensor_baseline = torch.empty(num_elems, dtype=dtype, device=device)

        # Prepare NVRAR tensor if available
        nvrar_tensor = None
        nvrar_tensor_id = None
        algorithm = "recursive"
        if comm_wrapper is not None:
            # Allocate symmetric tensor via nvshmem4py and register with wrapper
            nvrar_tensor = nvshmem.tensor((num_elems,), dtype=dtype)
            nvrar_tensor_id = comm_wrapper.register_tensor(nvrar_tensor, nvshmem_comm_cuda.Protocol.LL8)

            # Choose kernel params
            if params_resolver is not None:
                params = params_resolver.for_message_bytes(num_elems * dtype_size)
                nb = int(params.get("num_blocks", 8))
                tpb = int(params.get("threads_per_block", 256))
                cb = int(params.get("chunk_bytes", 16384))
                algorithm = str(params.get("algorithm", "recursive"))
            else:
                nb, tpb, cb = 8, 256, 16384
            comm_wrapper.set_kernel_params(nvshmem_comm_cuda.Protocol.LL8, nb, tpb, cb)

        dist.barrier()

        # Baseline benchmark (with optional CUDA Graphs)
        if args.graphs:
            try:
                avg_ms_baseline_evt, avg_ms_baseline_mpi = benchmark_torch_allreduce_graph(
                    tensor_baseline, args.iterations, args.warmup, stream, process_group=None, inner_iters=args.graph_inner_iters, use_matmul=args.graph_use_matmul
                )
            except Exception as e:
                if rank == 0 and not args.quiet:
                    print(f"CUDA Graphs baseline capture failed, falling back: {e}")
                avg_ms_baseline_evt, avg_ms_baseline_mpi = benchmark_torch_allreduce(
                    tensor_baseline, args.iterations, args.warmup, stream, process_group=None
                )
        else:
            avg_ms_baseline_evt, avg_ms_baseline_mpi = benchmark_torch_allreduce(
                tensor_baseline, args.iterations, args.warmup, stream, process_group=None
            )

        avg_ms_nvrar_evt = None
        avg_ms_nvrar_mpi = None
        if comm_wrapper is not None:
            if args.graphs:
                try:
                    avg_ms_nvrar_evt, avg_ms_nvrar_mpi = benchmark_nvrar_allreduce_graph(
                        comm_wrapper, nvrar_tensor, nvrar_tensor_id, args.iterations, args.warmup, stream, algorithm, stream_ptr, process_group=None, inner_iters=args.graph_inner_iters, use_matmul=args.graph_use_matmul
                    )
                except Exception as e:
                    if rank == 0 and not args.quiet:
                        print(f"CUDA Graphs NVRAR capture failed, falling back: {e}")
                    avg_ms_nvrar_evt, avg_ms_nvrar_mpi = benchmark_nvrar_allreduce(
                        comm_wrapper, nvrar_tensor, nvrar_tensor_id, args.iterations, args.warmup, stream, algorithm, stream_ptr, process_group=None
                    )
            else:
                avg_ms_nvrar_evt, avg_ms_nvrar_mpi = benchmark_nvrar_allreduce(
                    comm_wrapper, nvrar_tensor, nvrar_tensor_id, args.iterations, args.warmup, stream, algorithm, stream_ptr, process_group=None
                )

        # Correctness check (single check per method)
        world = world_size
        baseline_ok = True
        with torch.cuda.stream(stream):
            tensor_baseline.fill_(1)
            dist.all_reduce(tensor_baseline, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(device=device)
        expected = torch.ones(num_elems, dtype=dtype, device=device) * world
        if tensor_baseline.dtype in (torch.float16, torch.bfloat16):
            baseline_ok = bool(torch.allclose(tensor_baseline, expected, rtol=1e-2, atol=1e-2))
        elif tensor_baseline.dtype == torch.float32:
            baseline_ok = bool(torch.allclose(tensor_baseline, expected, rtol=1e-4, atol=1e-5))
        else:
            baseline_ok = bool(torch.equal(tensor_baseline, expected))

        nvrar_ok = None
        if comm_wrapper is not None:
            with torch.cuda.stream(stream):
                nvrar_tensor.fill_(1)
                comm_wrapper.allreduce_preallocated(nvrar_tensor, nvrar_tensor_id, stream_ptr, algorithm)
            torch.cuda.synchronize(device=device)
            if nvrar_tensor.dtype in (torch.float16, torch.bfloat16):
                nvrar_ok = bool(torch.allclose(nvrar_tensor, expected, rtol=1e-2, atol=1e-2))
            elif nvrar_tensor.dtype == torch.float32:
                nvrar_ok = bool(torch.allclose(nvrar_tensor, expected, rtol=1e-4, atol=1e-5))
            else:
                nvrar_ok = bool(torch.equal(nvrar_tensor, expected))

        # Aggregate correctness across ranks
        ok_tensor = torch.tensor([1 if baseline_ok else 0], device=device, dtype=torch.int)
        dist.all_reduce(ok_tensor, op=dist.ReduceOp.MIN)
        baseline_all_ok = (int(ok_tensor.item()) == 1)

        if comm_wrapper is not None:
            ok_tensor2 = torch.tensor([1 if (nvrar_ok is True) else 0], device=device, dtype=torch.int)
            dist.all_reduce(ok_tensor2, op=dist.ReduceOp.MIN)
            nvrar_all_ok = (int(ok_tensor2.item()) == 1)
        else:
            nvrar_all_ok = None

        if rank == 0:
            size_bytes = num_elems * dtype_size
            print(f"Size={size_bytes} bytes | dtype={args.dtype}")
            print(f"  torch.all_reduce: {avg_ms_baseline_evt:.4f} ms (evt) | {avg_ms_baseline_mpi:.4f} ms (mpi) [OK={baseline_all_ok}]")
            if avg_ms_nvrar_evt is not None:
                print(f"  nvrar (NVSHMEM): {avg_ms_nvrar_evt:.4f} ms (evt) | {avg_ms_nvrar_mpi:.4f} ms (mpi) [OK={nvrar_all_ok}] (algo={algorithm})")

        dist.barrier()

        # Cleanup
        if comm_wrapper is not None and nvrar_tensor_id is not None:
            comm_wrapper.deregister_tensor(nvrar_tensor_id)
            nvshmem.free_tensor(nvrar_tensor)

    dist.barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()


