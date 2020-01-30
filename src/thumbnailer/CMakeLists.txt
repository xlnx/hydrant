file(GLOB_RECURSE SOURCES *.cc)

cuda_add_executable(thumbnailer ${SOURCES})

target_include_directories(thumbnailer PUBLIC
  ${PROJECT_SOURCE_DIR}/include
)

vm_target_dependency(thumbnailer voxel_unarchive PUBLIC)
vm_target_dependency(thumbnailer cudafx PUBLIC)
