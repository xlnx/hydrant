if (HYDRA_BUILD_ENGINE)
  add_subdirectory(engine)
endif()

if (HYDRA_BUILD_GUI)
  add_subdirectory(gui)
endif()

add_subdirectory(renderers)

if (HYDRA_BUILD_LOCAL)
  add_subdirectory(local)
endif()

if (HYDRA_BUILD_SERVER)
  add_subdirectory(server)
endif()
