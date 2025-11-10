cmake_minimum_required(VERSION 3.17)
project(nvshmem_comm_cuda LANGUAGES CXX CUDA)

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures")
endif()

# 1) Find Python interpreter + development headers (for <Python.h>)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# 2) Find Torch
find_package(Torch QUIET CONFIG)

if(NOT Torch_FOUND)
  message(STATUS "Torch not found via default paths; trying Python hint…")
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
      "import torch, os; print(os.path.dirname(torch.__file__))"
    OUTPUT_VARIABLE TORCH_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  message(STATUS "Detected Torch root at: ${TORCH_ROOT}")

  # Tell CMake to search TorchConfig.cmake under Torch/share/cmake,
  # and also to look for libraries under Torch/lib
  list(APPEND CMAKE_PREFIX_PATH   "${TORCH_ROOT}")
  list(APPEND CMAKE_LIBRARY_PATH  "${TORCH_ROOT}/lib")
  find_package(Torch REQUIRED CONFIG)
endif()

# 3) Find MPI (for mpi.h + MPI::MPI_CXX)
find_package(MPI REQUIRED COMPONENTS CXX)

# 4) Pick up NVSHMEM_HOME from environment (or via -DNVSHMEM_ROOT)
if(NOT DEFINED NVSHMEM_ROOT)
  if(DEFINED ENV{NVSHMEM_HOME})
    set(NVSHMEM_ROOT $ENV{NVSHMEM_HOME})
  else()
    message(FATAL_ERROR
      "NVSHMEM_ROOT not set. Either export NVSHMEM_HOME or pass -DNVSHMEM_ROOT=<path>.")
  endif()
endif()
message(STATUS "Using NVSHMEM_ROOT = ${NVSHMEM_ROOT}")

# 5) Build exactly one Python extension module
add_library(nvshmem_comm_cuda MODULE
    src/nvshmem_comm.cpp
    src/ll8_coll.cu
    src/simple_coll.cu
    src/nvshmem_comm_bindings.cpp   
)

# define TORCH_EXTENSION_NAME so pybind11 sees the right name:
target_compile_definitions(nvshmem_comm_cuda PRIVATE
  TORCH_EXTENSION_NAME=nvshmem_comm_cuda
)

set_target_properties(nvshmem_comm_cuda PROPERTIES
  PREFIX ""                              # no “lib” prefix
  SUFFIX ".so"                           # .so suffix for Python
  LINKER_LANGUAGE CUDA                   # let nvcc drive the link
  CXX_STANDARD 17
  CUDA_SEPARABLE_COMPILATION ON          # for NVSHMEM device calls
)

# 9) Include paths for Python, Torch, MPI, NVSHMEM
target_include_directories(nvshmem_comm_cuda PRIVATE
  ${Python3_INCLUDE_DIRS}                             # <Python.h>
  ${TORCH_INCLUDE_DIRS}                               # Torch C++ API
  ${MPI_CXX_INCLUDE_DIRS}                             # mpi.h
  ${NVSHMEM_ROOT}/include                             # NVSHMEM headers
  ${CMAKE_CURRENT_SOURCE_DIR}/include                 # Our headers
)

# 9.1) CUDA 13 CCCL headers live under <CUDA_ROOT>/include/cccl
# Try to locate CUDA root from CUDA_HOME/CUDA_PATH or from nvcc
set(_CUDA_ROOT "")
if(DEFINED ENV{CUDA_HOME})
  set(_CUDA_ROOT $ENV{CUDA_HOME})
elseif(DEFINED ENV{CUDA_PATH})
  set(_CUDA_ROOT $ENV{CUDA_PATH})
elseif(CMAKE_CUDA_COMPILER)
  get_filename_component(_CUDA_BIN_DIR "${CMAKE_CUDA_COMPILER}" DIRECTORY)
  get_filename_component(_CUDA_ROOT "${_CUDA_BIN_DIR}" DIRECTORY)
endif()
if(_CUDA_ROOT AND EXISTS "${_CUDA_ROOT}/include/cccl")
  message(STATUS "Adding CUDA CCCL include path: ${_CUDA_ROOT}/include/cccl")
  target_include_directories(nvshmem_comm_cuda PRIVATE "${_CUDA_ROOT}/include/cccl")
endif()

# Torch Python shim: prefer target if present, else raw library
set(_need_torch_python ON)
if(TARGET Torch::Python)
  target_link_libraries(nvshmem_comm_cuda PRIVATE Torch::Python)
  set(_need_torch_python OFF)
endif()

if(_need_torch_python)
  # Older TorchConfig variants don’t export Torch::Python
  find_library(TORCH_PYTHON_LIBRARY
    NAMES torch_python torch_python3
    HINTS "${TORCH_INSTALL_PREFIX}/lib" "${TORCH_ROOT}/lib"
  )
  if(TORCH_PYTHON_LIBRARY)
    message(INFO " Found Torch python library")
    target_link_libraries(nvshmem_comm_cuda PRIVATE ${TORCH_PYTHON_LIBRARY})
  else()
    message(WARNING "Could not find torch_python; import errors likely.")
  endif()
endif()


# 10) Link against Torch, Torch’s Python binding lib, CUDA, NVSHMEM, MPI
target_link_libraries(nvshmem_comm_cuda PRIVATE
  ${TORCH_LIBRARIES}                     # libtorch.so, libc10.so, …
  Python3::Module
  CUDA::cudart
  ${NVSHMEM_ROOT}/lib/libnvshmem_host.so
  ${NVSHMEM_ROOT}/lib/libnvshmem_device.a
  MPI::MPI_CXX                           # MPI C++ library
)
if(TARGET Torch::Python)
  target_link_libraries(nvshmem_comm_cuda PRIVATE Torch::Python)
endif()

# 11) At runtime look next to the .so for libc10.so, libmpi, etc.
set_target_properties(nvshmem_comm_cuda PROPERTIES
  INSTALL_RPATH "$ORIGIN"
)

# Install the Python extension into the package directory inside the wheel
install(TARGETS nvshmem_comm_cuda
  LIBRARY DESTINATION nvrar
)


