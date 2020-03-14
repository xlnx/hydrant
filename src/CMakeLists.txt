if (HYDRA_BUILD_ENGINE)
  add_subdirectory(engine)
endif()

if (HYDRA_BUILD_GUI)
  add_subdirectory(gui)
endif()

if (HYDRA_BUILD_LOCAL)
  add_subdirectory(local)
endif()
