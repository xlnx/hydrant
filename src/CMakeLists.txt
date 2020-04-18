if (HYDRA_BUILD_SERVER)
  add_subdirectory(server)
  add_subdirectory(zk)
endif()

if (HYDRA_BUILD_CLIENT)
  add_subdirectory(client)
endif()
