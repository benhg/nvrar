// Copyright 2025 Parallel Software and Systems Group, University of Maryland.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once

#include <cuda_runtime.h>
#include <torch/extension.h>

#include <nvshmem.h>
#include <nvshmemx.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <string>
#include <tuple>

#include "nvshmem_utils.h"
#include "coll_factory.h"

class NVSHMEMCommWrapper {
 public:
  NVSHMEMCommWrapper(int rank, int world_size, int device);
  ~NVSHMEMCommWrapper();

  // Disable copy constructor and assignment operator
  NVSHMEMCommWrapper(const NVSHMEMCommWrapper&) = delete;
  NVSHMEMCommWrapper& operator=(const NVSHMEMCommWrapper&) = delete;

  void destroy();

  // Register an externally-allocated symmetric tensor (e.g., nvshmem4py)
  // Returns an internal tensor id used by collectives
  uint64_t register_tensor(torch::Tensor& tensor, Protocol protocol);
  void deregister_tensor(uint64_t id);

  // Collective operations
  void allreduce_preallocated(torch::Tensor& tensor, uint64_t id,
                              uint64_t stream_ptr,
                              std::string alg = "recursive");

  // Configuration methods
  void set_kernel_params(Protocol protocol, int num_blocks,
                         int threads_per_block, size_t chunk_size);

 private:
  void initialize_coll(Protocol protocol);

  int rank_;
  int world_size_;
  int mype_;
  int npes_;
  int device_;
  bool initialized_;
  bool owns_nvshmem_init_ = false;

  nvshmemx_uniqueid_t uid_ = NVSHMEMX_UNIQUEID_INITIALIZER;

  std::unordered_map<uint64_t, Protocol> tensor_to_protocol_map_;
  std::unordered_map<Protocol, std::unique_ptr<IColl>> coll_map_;
};
