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
vm_external_module(
  GIT_REPOSITORY https://github.com/cad420/glm-cuda
  GIT_TAG        master
)
vm_external_module(
  GIT_REPOSITORY https://github.com/cad420/cppfs
  GIT_TAG        master
)

#find_package(cpprestsdk REQUIRED)
find_package(MPI REQUIRED)

file(GLOB_RECURSE SOURCES *.cc *.cu *.c)

cuda_add_executable(hydra-server ${SOURCES})
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
target_link_libraries(hydra-server ${MPI_CXX_LIBRARIES})

vm_target_dependency(hydra-server VMat PUBLIC)
vm_target_dependency(hydra-server VMUtils PUBLIC)
vm_target_dependency(hydra-server voxel_unarchive PUBLIC)
vm_target_dependency(hydra-server cudafx PUBLIC)
vm_target_dependency(hydra-server glm-cuda PUBLIC)
vm_target_dependency(hydra-server cppfs PUBLIC)
target_link_libraries(hydra-server glog)

install(TARGETS hydra-server
  RUNTIME DESTINATION .
)
