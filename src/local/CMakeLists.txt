file(GLOB_RECURSE SOURCES *.cc)

cuda_add_executable(hydra-local ${SOURCES})

vm_target_dependency(hydra-local hydra-engine PUBLIC)
vm_target_dependency(hydra-local hydra-gui PUBLIC)
target_link_libraries(hydra-local 
    -Wl,--whole-archive 
    hydra-renderers hydra-renderer-uis 
    -Wl,--no-whole-archive
)

install(TARGETS hydra-local
    RUNTIME DESTINATION .
)