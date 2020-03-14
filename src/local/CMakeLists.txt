file(GLOB_RECURSE SOURCES *.cc)

cuda_add_executable(hydra-local ${SOURCES})

vm_target_dependency(hydra-local hydra-engine PUBLIC)
vm_target_dependency(hydra-local hydra-gui PUBLIC)
target_link_libraries(hydra-local 
    hydra-engine
    hydra-gui 
)
install(TARGETS hydra-local
    RUNTIME DESTINATION .
)