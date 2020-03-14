cuda_add_library(isosurface.mod SHARED
    isosurface.cc
    isosurface_shader.cu
)
vm_target_dependency(isosurface.mod hydra-engine PRIVATE)
install_mod(isosurface.mod)

cuda_add_library(isosurface.ui SHARED
    isosurface.ui.cc
)
vm_target_dependency(isosurface.ui hydra-gui PRIVATE)
install_mod(isosurface.ui)
