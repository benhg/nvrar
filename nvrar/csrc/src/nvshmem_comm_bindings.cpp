// Copyright 2025 Parallel Software and Systems Group, University of Maryland.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>

#include "coll.h"
#include "nvshmem_comm.h"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Expose the Protocol enum
  py::enum_<Protocol>(m, "Protocol")
      .value("SIMPLE", Protocol::SIMPLE)
      .value("LL8", Protocol::LL8);

  py::class_<NVSHMEMCommWrapper, std::shared_ptr<NVSHMEMCommWrapper>>(
      m, "NVSHMEMCommWrapper")
      .def(py::init<int, int, int>())
      .def("destroy", &NVSHMEMCommWrapper::destroy)
      .def("allreduce_preallocated",
           &NVSHMEMCommWrapper::allreduce_preallocated)
      .def("register_tensor", &NVSHMEMCommWrapper::register_tensor)
      .def("deregister_tensor", &NVSHMEMCommWrapper::deregister_tensor)
      .def("set_kernel_params", &NVSHMEMCommWrapper::set_kernel_params);
}
