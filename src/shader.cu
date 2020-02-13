#include <hydrant/shader.hpp>

VM_BEGIN_MODULE( hydrant )

__constant__ char shader_args_buffer[ 1024 * 16 ];

/* Ray Emit Kernel Impl */

void ShaderDesc::copy_to_buffer( const void *udata, std::size_t size ) const
{
  cudaMemcpyToSymbol( shader_args_buffer, udata, size, offset );
}

__global__ void
  ray_emit_kernel_impl( RayEmitKernelArgs args )
{
  uint x = blockIdx.x * blockDim.x + threadIdx.x;
  uint y = blockIdx.y * blockDim.y + threadIdx.y;
  
  if ( x >= args.image.resolution.x || y >= args.image.resolution.y ) {
    return;
	}
  
  auto cc = vec2( args.image.resolution ) / 2.f;
  auto uv = ( vec2{ x, y } - cc ) * 2.f / float( args.image.resolution.y );
  Ray ray = { 
    args.view.ray_o, 
    normalize( vec3( args.view.trans * vec4( uv.x, -uv.y, -args.view.itg_fovy, 1 ) ) - args.view.ray_o )
  };

  auto shader = ( p_ray_emit_shader_t )args.shader.shader;
  shader( ray, 
          args.image.data + args.image.pixel_size * ( args.image.resolution.x * y + x ), 
          shader_args_buffer + args.shader.offset );
}

CUFX_DEFINE_KERNEL( ray_emit_kernel, ray_emit_kernel_impl );

/* Pixel Kernel Impl */

__global__ void
  pixel_kernel_impl( PixelKernelArgs args )
{
  uint x = blockIdx.x * blockDim.x + threadIdx.x;
  uint y = blockIdx.y * blockDim.y + threadIdx.y;
  
  if ( x >= args.image.resolution.x || y >= args.image.resolution.y ) {
    return;
	}

  auto shader = ( p_pixel_shader_t )args.shader.shader;
  shader( args.image.data + args.image.pixel_size * ( args.image.resolution.x * y + x ), 
          shader_args_buffer + args.shader.offset );
}

CUFX_DEFINE_KERNEL( pixel_kernel, pixel_kernel_impl );

VM_END_MODULE()
