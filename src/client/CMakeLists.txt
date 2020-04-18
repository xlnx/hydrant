vm_external_module(
  GIT_REPOSITORY https://github.com/cad420/VMUtils
  GIT_TAG        master
)
vm_external_module(
  GIT_REPOSITORY https://github.com/cad420/glm-cuda
  GIT_TAG        master
)
vm_external_module(
  GIT_REPOSITORY https://github.com/xlnx/imgui-gl
  GIT_TAG        master
)

file(GLOB_RECURSE SOURCES *.c *.cc)

cuda_add_executable(hydra-client ${SOURCES})

vm_target_dependency(hydra-client VMUtils PUBLIC)
vm_target_dependency(hydra-client glm-cuda PUBLIC)
vm_target_dependency(hydra-client imgui PUBLIC)
vm_target_dependency(hydra-client cudafx PUBLIC)
vm_target_dependency(hydra-client cppfs PUBLIC)
vm_target_dependency(hydra-client voxel_unarchive PUBLIC)

#vm_target_dependency(hydra-server VMat PUBLIC)
#vm_target_dependency(hydra-server VMUtils PUBLIC)
#vm_target_dependency(hydra-server voxel_unarchive PUBLIC)
#vm_target_dependency(hydra-server cudafx PUBLIC)
#vm_target_dependency(hydra-server glm-cuda PUBLIC)
#vm_target_dependency(hydra-server cppfs PUBLIC)
#TODO: FIX DEPENDENCY ERROR

target_link_libraries(hydra-client glfw glog dl cpprest boost_system)

install(TARGETS hydra-client
  RUNTIME DESTINATION .
)

