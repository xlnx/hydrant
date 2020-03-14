cuda_add_library(paging.mod SHARED 
    paging.cc
    paging_shader.cu
)
vm_target_dependency(paging.mod hydra-engine PRIVATE)
install_mod(paging.mod)
