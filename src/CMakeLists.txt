#if (HYDRA_BUILD_SERVER)
#  add_subdirectory(zookeeper)
#  add_subdirectory(slave)
#endif()

#if (HYDRA_BUILD_CLIENT)
#  add_subdirectory(client)
#endif()

vm_external_module(
  GIT_REPOSITORY https://github.com/cad420/varch
  GIT_TAG        dev/flingza
)
