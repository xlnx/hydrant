cmake_minimum_required(VERSION 3.0)
project(hydrant)
set(CMAKE_CXX_STANDARD 14)

find_package(Git)
execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive)
# install external modules
include(scripts/external.cmake)

vm_external_module(
  GIT_REPOSITORY https://github.com/cad420/VMUtils
  GIT_TAG        master
)
vm_external_module(
  GIT_REPOSITORY https://github.com/cad420/varch
  GIT_TAG        dev/flingza
)
vm_external_module(
  GIT_REPOSITORY https://github.com/cad420/cuda-fx
  GIT_TAG        master
)

find_package(CUDA)
set(CUDA_NVCC_FLAGS ${CMAKE_NVCC_FLAGS}
  -Xcudafe
  --diag_suppress="2913" 
)

add_subdirectory(src)
