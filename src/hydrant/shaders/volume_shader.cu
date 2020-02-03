#include "volume_shader.hpp"

VM_BEGIN_MODULE(hydrant)

VM_EXPORT
{
    SHADER_IMPL( VolumnRayEmitShader<Raymarcher> );
}

VM_END_MODULE()
