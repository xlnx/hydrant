cuda_add_library(volume.mod SHARED
    volume.cc
    volume_shader.cu
)
vm_target_dependency(volume.mod hydra-engine PRIVATE)
install_mod(volume.mod)

cuda_add_library(volume.ui SHARED
    volume.ui.cc
)
vm_target_dependency(volume.ui hydra-gui PRIVATE)
install_mod(volume.ui)
