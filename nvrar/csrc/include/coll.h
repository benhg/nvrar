// Copyright 2025 Parallel Software and Systems Group, University of Maryland.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <torch/extension.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <string>
#include <tuple>

enum class Protocol : uint8_t {
  SIMPLE,
  LL8,
};

class IColl {
 public:
  virtual ~IColl() noexcept = default;

  virtual void init(int num_blocks, int threads_per_block,
                    size_t chunk_size) = 0;

  // Register an externally-allocated symmetric tensor (e.g., via nvshmem4py)
  // Returns a newly assigned tensor id
  virtual uint64_t register_external_tensor(torch::Tensor& t) = 0;
  // Deregister a previously registered tensor without freeing memory
  virtual void deregister_tensor(uint64_t id) = 0;

  virtual void dispatch_allreduce_preallocated(torch::Tensor& t, uint64_t id,
                                               cudaStream_t s,
                                               const std::string& alg) = 0;

  virtual void set_kernel_params(int num_blocks, int threads_per_block,
                                 size_t chunk_size) = 0;
};

// Abstract base class for protocol-specific collective operations
template <class Derived>
class CollBase : public IColl {
 public:
  ~CollBase() noexcept override {
    // Nothing to free: external tensors are owned by nvshmem4py
  }

  void init(int num_blocks, int threads_per_block, size_t chunk_size) override {
    // Check if nvshmem is initialized
    // TODO: Currently MPG is not supported
    if (nvshmemx_init_status() != NVSHMEM_STATUS_IS_INITIALIZED) {
      throw std::runtime_error("NVSHMEM is not initialized");
    }
    mype_ = nvshmem_my_pe();
    npes_ = nvshmem_n_pes();
    mype_node_ = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    npes_node_ = nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE);

    derived()->initialize(num_blocks, threads_per_block, chunk_size);
  }

  uint64_t register_external_tensor(torch::Tensor& t) override {
    // Accept a pre-allocated symmetric tensor; we do not own its memory
    void* ptr = t.data_ptr();
    if (ptr == nullptr) {
      throw std::runtime_error("register_external_tensor: null data_ptr");
    }
    uint64_t id = next_id_.fetch_add(1);
    // Let derived class register scratch/meta using size/dtype/device
    const size_t size = static_cast<size_t>(t.numel());
    const torch::Dtype dt = t.scalar_type();
    const torch::Device dev = t.device();
    derived()->register_tensor(id, size, dt, dev);
    return id;
  }

  void deregister_tensor(uint64_t id) override {
    derived()->deregister_tensor(id);
  }

  void dispatch_allreduce_preallocated(torch::Tensor& t, uint64_t id,
                                       cudaStream_t s,
                                       const std::string& alg) override {
    auto dtype = t.dtype();
    if (dtype == torch::kFloat32) {
      derived()->template allreduce_preallocated_impl<float>(t, id, s, alg);
    } else if (dtype == torch::kFloat16) {
      derived()->template allreduce_preallocated_impl<__half>(t, id, s, alg);
    } else if (dtype == torch::kBFloat16) {
      derived()->template allreduce_preallocated_impl<__nv_bfloat16>(t, id, s,
                                                                     alg);
    } else if (dtype == torch::kInt32) {
      derived()->template allreduce_preallocated_impl<int>(t, id, s, alg);
    } else {
      throw std::runtime_error("Unsupported tensor dtype for allreduce");
    }
  }

  void set_kernel_params(int num_blocks, int threads_per_block,
                         size_t chunk_size) override {
    derived()->set_kernel_params(num_blocks, threads_per_block, chunk_size);
  }

 private:
  Derived* derived() { return static_cast<Derived*>(this); }

 protected:
  // Next ID for allocated tensors
  std::atomic<uint64_t> next_id_;

  // PE information
  int mype_;
  int npes_;
  int mype_node_;
  int npes_node_;
};
