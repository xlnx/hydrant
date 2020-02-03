#include "raycaster.hpp"

VM_BEGIN_MODULE( hydrant )

__constant__ char shader_args_buffer[ 4096 ];

void ShaderDesc::copy_to_buffer( const void *udata, std::size_t size ) const
{
  cudaMemcpyToSymbol( shader_args_buffer, udata, size, offset );
}

__global__ void
  cast_kernel_impl( RayEmitterArguments args )
{
  uint x = blockIdx.x * blockDim.x + threadIdx.x;
  uint y = blockIdx.y * blockDim.y + threadIdx.y;
  
  if ( x >= args.resolution.x || y >= args.resolution.y ) {
    return;
	}
  
  auto cc = vec2( args.resolution ) / 2.f;
  auto uv = ( vec2{ x, y } - cc ) * 2.f / float( args.resolution.y );
  Ray ray = { 
    args.ray_o, 
    normalize( vec3( args.trans * vec4( uv.x, -uv.y, -args.itg_fovy, 1 ) ) - args.ray_o )
  };

  args.shader( ray, 
               args.image + args.pixel_size * ( args.resolution.x * y + x ), 
               shader_args_buffer + args.shader_args_offset );
}

CUFX_DEFINE_KERNEL( cast_kernel, cast_kernel_impl );

VM_END_MODULE()
