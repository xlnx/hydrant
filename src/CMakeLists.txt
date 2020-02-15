file(GLOB_RECURSE SOURCES *.cc *.cu)

cuda_add_executable(hydrant ${SOURCES})

target_include_directories(hydrant PUBLIC
  ${PROJECT_SOURCE_DIR}/include
)

vm_target_dependency(hydrant voxel_unarchive PUBLIC)
vm_target_dependency(hydrant cudafx PUBLIC)
vm_target_dependency(hydrant cppfs PUBLIC)
vm_target_dependency(hydrant glm-cuda PUBLIC)
