vm_external_module(
  GIT_REPOSITORY https://github.com/cad420/VMUtils
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

cuda_add_executable(hydra-zk ${SOURCES})
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
target_link_libraries(hydra-zk ${MPI_CXX_LIBRARIES})

target_link_libraries(hydra-zk cpprest glog boost_system)
vm_target_dependency(hydra-zk VMUtils PUBLIC)
vm_target_dependency(hydra-zk glm-cuda PUBLIC)
vm_target_dependency(hydra-zk cppfs PUBLIC)

install(TARGETS hydra-zk
  RUNTIME DESTINATION .
)
install(FILES launch_server.sh hostfile.example
  DESTINATION .
)
