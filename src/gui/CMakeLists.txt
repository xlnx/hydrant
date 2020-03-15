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

cuda_add_library(hydra-gui ${SOURCES})

vm_target_dependency(hydra-gui VMUtils PUBLIC)
vm_target_dependency(hydra-gui glm-cuda PUBLIC)
vm_target_dependency(hydra-gui imgui PUBLIC)

target_link_libraries(hydra-gui glfw)
target_link_libraries(hydra-gui dl)
