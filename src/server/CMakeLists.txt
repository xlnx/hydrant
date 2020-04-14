file(GLOB_RECURSE SOURCES *.cc)

cuda_add_executable(hydra-server ${SOURCES})

vm_target_dependency(hydra-server hydra-engine PUBLIC)
target_link_libraries(hydra-server 
    -Wl,--whole-archive 
    hydra-renderers
    -Wl,--no-whole-archive
)

install(TARGETS hydra-server
    RUNTIME DESTINATION .
)
