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

file(GLOB_RECURSE SOURCES src/*.cc)

cuda_add_executable(hydrant ${SOURCES})

target_include_directories(hydrant PUBLIC
  ${PROJECT_SOURCE_DIR}/include
)

vm_target_dependency(hydrant voxel_unarchive PUBLIC)
vm_target_dependency(hydrant cudafx PUBLIC)
