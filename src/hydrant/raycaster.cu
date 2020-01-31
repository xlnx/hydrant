#include "raycaster.hpp"

VM_BEGIN_MODULE( hydrant )

__constant__ char argument_buffer[ 4096 ];

void copy_to_argument_buffer( std::size_t offset, const void *udata, std::size_t size )
{
  cudaMemcpyToSymbol( argument_buffer, udata, size, offset );
}

__global__ void
  cast_kernel_impl( CastOptions opts )
{
  uint x = blockIdx.x * blockDim.x + threadIdx.x;
  uint y = blockIdx.y * blockDim.y + threadIdx.y;
  
  if ( x >= opts.resolution.x || y >= opts.resolution.y ) {
    return;
	}
  
  auto cc = vec2( opts.resolution ) / 2.f;
  auto uv = ( vec2{ x, y } - cc ) * 2.f / float( opts.resolution.y );
  Ray ray = { 
    opts.ray_o, 
    normalize( vec3( opts.trans * vec4( uv.x, -uv.y, -opts.itg_fovy, 1 ) ) - opts.ray_o )
  };

  opts.shader( ray, 
               opts.image + opts.pixel_size * ( opts.resolution.x * y + x ), 
               argument_buffer + opts.udata_offset );
}

CUFX_DEFINE_KERNEL( cast_kernel, cast_kernel_impl );

VM_END_MODULE()
