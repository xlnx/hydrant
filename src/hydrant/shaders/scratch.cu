#include "scratch.hpp"

VM_BEGIN_MODULE( hydrant )

VM_EXPORT 
{
    SHADER_IMPL( ChebyshevShader<ScratchIntegrator> );
}

VM_END_MODULE()
