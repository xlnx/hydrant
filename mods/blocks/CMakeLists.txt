cuda_add_library(blocks.mod SHARED
    blocks.cc
    blocks_shader.cu
)
vm_target_dependency(blocks.mod hydra-engine PRIVATE)
install_mod(blocks.mod)

cuda_add_library(blocks.ui SHARED
    blocks.ui.cc
)
vm_target_dependency(blocks.ui hydra-gui PRIVATE)
install_mod(blocks.ui)
