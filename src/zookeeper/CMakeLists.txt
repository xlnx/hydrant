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

cuda_add_executable(hydra-zookeeper ${SOURCES})
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
target_link_libraries(hydra-zookeeper ${MPI_CXX_LIBRARIES})

target_link_libraries(hydra-zookeeper cpprest glog boost_system)
vm_target_dependency(hydra-zookeeper VMUtils PUBLIC)
vm_target_dependency(hydra-zookeeper glm-cuda PUBLIC)
vm_target_dependency(hydra-zookeeper cppfs PUBLIC)

install(TARGETS hydra-zookeeper
  RUNTIME DESTINATION .
)
install(FILES launch_server.sh hostfile.example
  DESTINATION .
)
