#include <hydrant/core/shader.hpp>

VM_BEGIN_MODULE( hydrant )

__constant__ char shader_args_buffer[ 1024 * 16 ];

void DeviceFunctionDesc::copy_to_buffer( const void *udata, std::size_t size ) const
{
	cudaMemcpyToSymbol( shader_args_buffer, udata, size, offset );
}

/* Ray Emit Kernel Impl */

__global__ void
  ray_emit_kernel_impl( CudaRayEmitKernelArgs args )
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;

	if ( x >= args.image_desc.resolution.x || y >= args.image_desc.resolution.y ) {
		return;
	}

	auto cc = vec2( args.image_desc.resolution ) / 2.f;
	auto uv = ( vec2{ x, y } - cc ) * 2.f / float( args.image_desc.resolution.y );
	Ray ray = {
		args.view.ray_o,
		normalize( vec3( args.view.trans * vec4( uv.x, -uv.y, -args.view.ctg_fovy_2, 1 ) ) - args.view.ray_o )
	};

	auto shader = (ray_emit_shader_t *)args.function_desc.fp;
	shader( ray,
			args.image_desc.data + args.image_desc.pixel_size * ( args.image_desc.resolution.x * y + x ),
			shader_args_buffer + args.function_desc.offset );
}

CUFX_DEFINE_KERNEL( ray_emit_kernel, ray_emit_kernel_impl );

/* Ray March Kernel Impl */

__global__ void
  ray_march_kernel_impl( CudaRayMarchKernelArgs args )
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;

	if ( x >= args.image_desc.resolution.x || y >= args.image_desc.resolution.y ) {
		return;
	}

	auto shader = (ray_march_shader_t *)args.function_desc.fp;
	shader( args.image_desc.data + args.image_desc.pixel_size * ( args.image_desc.resolution.x * y + x ),
			shader_args_buffer + args.function_desc.offset );
}

CUFX_DEFINE_KERNEL( ray_march_kernel, ray_march_kernel_impl );

/* Pixel Kernel Impl */

__global__ void
  pixel_kernel_impl( CudaPixelKernelArgs args )
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;

	if ( x >= args.image_desc.resolution.x || y >= args.image_desc.resolution.y ) {
		return;
	}

	auto shader = (pixel_shader_t *)args.function_desc.fp;
	shader( args.image_desc.data + args.image_desc.pixel_size * ( args.image_desc.resolution.x * y + x ),
			args.dst_desc.data + args.dst_desc.pixel_size * ( args.dst_desc.resolution.x * y + x ) );
}

CUFX_DEFINE_KERNEL( pixel_kernel, pixel_kernel_impl );

VM_END_MODULE()
