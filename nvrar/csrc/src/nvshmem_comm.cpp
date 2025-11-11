// Copyright 2025 Parallel Software and Systems Group, University of Maryland.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "nvshmem_comm.h"

#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include <cstdint>
#include <cstring>
#include <ctime>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

NVSHMEMCommWrapper::NVSHMEMCommWrapper(int rank, int world_size, int device)
    : rank_(rank),
      world_size_(world_size),
      device_(device),
      initialized_(false) {
  // Set device
  CUDA_CHECK(cudaSetDevice(device_));

  // Require NVSHMEM to be already initialized (via nvshmem4py)
  if (nvshmemx_init_status() != NVSHMEM_STATUS_IS_INITIALIZED) {
    throw std::runtime_error("NVSHMEM is not initialized. Initialize via nvshmem4py before constructing NVSHMEMCommWrapper.");
  }

  // Get PE information
  mype_ = nvshmem_my_pe();
  npes_ = nvshmem_n_pes();

  // Verify rank consistency
  if (mype_ != rank_ || npes_ != world_size_) {
    throw std::runtime_error(
        "MPI rank/world_size mismatch with NVSHMEM PE info");
  }

  // Initialize default protocol
  initialize_coll(Protocol::LL8);

  initialized_ = true;
  std::cout << "NVSHMEM initialized for PE " << mype_ << " on " << npes_
            << " PEs" << std::endl;
}

NVSHMEMCommWrapper::~NVSHMEMCommWrapper() {
  if (initialized_) {
    std::cout << "NVSHMEMCommWrapper destructor called" << std::endl;
    destroy();
  }
}

void NVSHMEMCommWrapper::destroy() {
  if (initialized_) {
    std::cout << "NVSHMEMCommWrapper destroying" << std::endl;
    nvshmem_barrier_all();
    coll_map_.clear();  // This will automatically delete all unique_ptr objects
    initialized_ = false;
  }
}

void NVSHMEMCommWrapper::initialize_coll(Protocol protocol) {
  if (coll_map_.find(protocol) == coll_map_.end()) {
    // Create the coll object using factory
    coll_map_[protocol] =
        std::unique_ptr<IColl>(CollFactory::create_coll(protocol));

    // Initialize with default kernel parameters
    coll_map_[protocol]->init(32, 512, 262144);
  }
}

uint64_t NVSHMEMCommWrapper::register_tensor(torch::Tensor& tensor, Protocol protocol) {
  // Ensure the protocol-specific coll object exists
  initialize_coll(protocol);
  auto& coll = coll_map_[protocol];
  // Register external symmetric tensor and get an id
  uint64_t id = coll->register_external_tensor(tensor);
  tensor_to_protocol_map_[id] = protocol;
  return id;
}

void NVSHMEMCommWrapper::deregister_tensor(uint64_t id) {
  if (tensor_to_protocol_map_.find(id) == tensor_to_protocol_map_.end()) {
    throw std::runtime_error("Invalid tensor ID");
  }
  Protocol protocol = tensor_to_protocol_map_[id];
  tensor_to_protocol_map_.erase(id);
  auto& coll = coll_map_[protocol];
  coll->deregister_tensor(id);
}

void NVSHMEMCommWrapper::allreduce_preallocated(torch::Tensor& tensor,
                                                uint64_t id,
                                                uint64_t stream_ptr,
                                                std::string alg) {
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

  // Get the protocol for this tensor
  if (tensor_to_protocol_map_.find(id) == tensor_to_protocol_map_.end()) {
    throw std::runtime_error("Invalid tensor ID");
  }
  Protocol protocol = tensor_to_protocol_map_[id];

  // Get the appropriate coll object
  auto& coll = coll_map_[protocol];
  coll->dispatch_allreduce_preallocated(tensor, id, stream, alg);
}

void NVSHMEMCommWrapper::set_kernel_params(Protocol protocol, int num_blocks,
                                           int threads_per_block,
                                           size_t chunk_size) {
  // Apply kernel parameters to the specified protocol
  if (coll_map_.find(protocol) == coll_map_.end()) {
    throw std::runtime_error("Protocol is not initialized");
  }
  auto& coll = coll_map_[protocol];
  coll->set_kernel_params(num_blocks, threads_per_block, chunk_size);
}

