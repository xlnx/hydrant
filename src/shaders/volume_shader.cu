#include "volume_shader.hpp"

VM_BEGIN_MODULE(hydrant)

VM_EXPORT
{
    SHADER_IMPL( VolumeRayEmitShader );
    SHADER_IMPL( VolumePixelShader );
}

VM_END_MODULE()
