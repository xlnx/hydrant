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
  GIT_REPOSITORY https://github.com/xlnx/glog
  GIT_TAG        master
)
vm_external_module(
  GIT_REPOSITORY https://github.com/cad420/cppfs
  GIT_TAG        master
)

file(GLOB_RECURSE SOURCES *.cc *.cu *.c)

cuda_add_library(hydra-engine ${SOURCES})

vm_target_dependency(hydra-engine VMat PUBLIC)
vm_target_dependency(hydra-engine VMUtils PUBLIC)
vm_target_dependency(hydra-engine voxel_unarchive PUBLIC)
vm_target_dependency(hydra-engine cudafx PUBLIC)
vm_target_dependency(hydra-engine glm-cuda PUBLIC)
vm_target_dependency(hydra-engine glog PUBLIC)
vm_target_dependency(hydra-engine cppfs PUBLIC)
