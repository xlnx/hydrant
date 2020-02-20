#pragma once

#include <map>
#include <string>
#include <functional>
#include <typeindex>
#include <typeinfo>
#include <VMUtils/enum.hpp>
#include <hydrant/core/glm_math.hpp>
#include <cudafx/kernel.hpp>
#include <cudafx/image.hpp>

VM_BEGIN_MODULE( hydrant )

struct IShaderTypeErased
{
	Box3D bbox;
	float step;
	int max_steps = 500;
};

VM_EXPORT
{
	VM_ENUM( ShadingDevice,
			 Cpu, Cuda );

	VM_ENUM( ShadingResult,
			 Ok, Err );

	struct IPixel
	{
		/* object space ray */
		Ray ray;
		/* maximum march steps left */
		int nsteps;
	};

	template <typename P>
	struct IShader : IShaderTypeErased
	{
		using Pixel = P;
	};
}

enum ShadingPass
{
	RayEmit,
	RayMarch,
	Pixel
};

struct ViewArgs
{
	mat4 trans;
	float ctg_fovy_2;
	vec3 ray_o;
};

struct ImageDesc
{
	ivec2 resolution;
	size_t pixel_size;
	char *data;

	template <typename P>
	void create_from_img( cufx::ImageView<P> const &img, bool device )
	{
		resolution = ivec2{ img.width(), img.height() };
		pixel_size = sizeof( P );
		data = reinterpret_cast<char *>( device ? &img.at_device( 0, 0 ) : &img.at_host( 0, 0 ) );
	}
};

using function_ptr_t = void ( * )();

struct DeviceFunctionDesc
{
	function_ptr_t fp = nullptr;
	std::size_t offset = 0;

	void copy_to_buffer( const void *udata, std::size_t size ) const;
};

struct ShaderMeta
{
	std::map<ShadingDevice::_enumerated,
			 std::function<ShadingResult( void * )>>
	  devices;
	std::string name;
	std::string class_name;
};

struct ShaderRegistry
{
	static ShaderRegistry instance;

	std::map<std::type_index, ShaderMeta> meta;
};

/* ray emit shader */

template <typename P, typename F>
__host__ __device__ void
  ray_emit_shader_impl( Ray const &ray_in, void *pixel_out, void const *shader_in )
{
	P pixel = {};
	F const &shader = *reinterpret_cast<F const *>( shader_in );

	pixel.ray = ray_in;
	float tnear, tfar;
	if ( pixel.ray.intersect( shader.bbox, tnear, tfar ) ) {
		pixel.ray.o += pixel.ray.d * tnear;
		pixel.nsteps = min( shader.max_steps, int( ( tfar - tnear ) / shader.step ) );
		shader.main( pixel );
	} else {
		pixel.nsteps = 0;
	}

	*reinterpret_cast<P *>( pixel_out ) = pixel;
}

using ray_emit_shader_t = void( Ray const &, void *, void const * );

template <typename P, typename F>
__device__ ray_emit_shader_t *p_ray_emit_shader = ray_emit_shader_impl<P, F>;

/* ray march shader */

template <typename P, typename F>
__host__ __device__ void
  ray_march_shader_impl( void *pixel_in_out, void const *shader_in )
{
	auto &pixel = *reinterpret_cast<P *>( pixel_in_out );
	F const &shader = *reinterpret_cast<F const *>( shader_in );
	auto pixel_reg = pixel;

	shader.main( pixel_reg );

	pixel = pixel_reg;
}

using ray_march_shader_t = void( void *, void const * );

template <typename P, typename F>
__device__ ray_march_shader_t *p_ray_march_shader = ray_march_shader_impl<P, F>;

/* pixel shader */

template <typename P, typename F>
__host__ __device__ void
  pixel_shader_impl( void const *pixel_in, void *pixel_out )
{
	auto &pixel_in_ = *reinterpret_cast<P const *>( pixel_in );
	auto &pixel_out_ = *reinterpret_cast<uchar4 *>( pixel_out );

	pixel_in_.write_to( pixel_out_ );
}

using pixel_shader_t = void( void const *, void * );

template <typename P, typename F>
__device__ pixel_shader_t *p_pixel_shader = pixel_shader_impl<P, F>;

/* kernel args defination */

struct BasicKernelArgs
{
	ShadingPass shading_pass;
	ImageDesc image_desc;
};

struct BasicRayEmitKernelArgs : BasicKernelArgs
{
	ViewArgs view;
};

struct BasicRayMarchKernelArgs : BasicKernelArgs
{
};

struct BasicPixelKernelArgs : BasicKernelArgs
{
	ImageDesc dst_desc;
};

struct CpuKernelLauncher
{
	function_ptr_t launcher;
	IShaderTypeErased const *shader;
};

struct CpuRayEmitKernelArgs : BasicRayEmitKernelArgs, CpuKernelLauncher
{
};

struct CpuRayMarchKernelArgs : BasicRayMarchKernelArgs, CpuKernelLauncher
{
};

struct CpuPixelKernelArgs : BasicPixelKernelArgs, CpuKernelLauncher
{
};

struct CudaShadingKernelLauncher
{
	DeviceFunctionDesc function_desc;
};

struct CudaRayEmitKernelArgs : BasicRayEmitKernelArgs, CudaShadingKernelLauncher
{
};

struct CudaRayMarchKernelArgs : BasicRayMarchKernelArgs, CudaShadingKernelLauncher
{
};

struct CudaPixelKernelArgs : BasicPixelKernelArgs, CudaShadingKernelLauncher
{
};

struct CudaShadingArgs
{
	cufx::KernelLaunchInfo launch_info;
	BasicKernelArgs *kernel_args;
	IShaderTypeErased const *shader;
};

struct ThreadPoolInfo
{
	unsigned nthreads = 1;
};

struct CpuShadingArgs
{
	ThreadPoolInfo thread_pool_info;
	BasicKernelArgs *kernel_args;
	IShaderTypeErased const *shader;
};

extern cufx::Kernel<void( CudaRayEmitKernelArgs args )> ray_emit_kernel;
extern cufx::Kernel<void( CudaRayMarchKernelArgs args )> ray_march_kernel;
extern cufx::Kernel<void( CudaPixelKernelArgs args )> pixel_kernel;

extern void ray_emit_task_dispatch( ThreadPoolInfo const &thread_pool_info,
									CpuRayEmitKernelArgs const &args );
extern void ray_march_task_dispatch( ThreadPoolInfo const &thread_pool_info,
									 CpuRayMarchKernelArgs const &args );
extern void pixel_task_dispatch( ThreadPoolInfo const &thread_pool_info,
								 CpuPixelKernelArgs const &args );

struct ShaderRegistrar
{
	ShaderRegistrar( std::type_index const &type_index,
					 std::string const &class_name ) :
	  _type_index( type_index )
	{
		_meta.class_name = class_name;
	}

	ShaderRegistrar &name( std::string const &value )
	{
		_meta.name = value;
		return *this;
	}

	template <typename T>
	ShaderRegistrar &cuda()
	{
		_meta.devices[ ShadingDevice::Cuda ] =
		  []( void *args_ptr ) -> ShadingResult {
			auto &args = *reinterpret_cast<CudaShadingArgs *>( args_ptr );
			switch ( args.kernel_args->shading_pass ) {
#define HYDRANT_CUDA_SHADER_IMPL_PASS( Pass, Lower )                              \
	case ShadingPass::Pass: {                                                     \
		auto &kargs = *static_cast<Cuda##Pass##KernelArgs *>( args.kernel_args ); \
		cudaMemcpyFromSymbol( &kargs.function_desc.fp,                            \
							  p_##Lower##_shader<typename T::Pixel, T>,           \
							  sizeof( kargs.function_desc.fp ) );                 \
		kargs.function_desc.offset = 0;                                           \
		kargs.function_desc.copy_to_buffer( args.shader, sizeof( T ) );           \
		Lower##_kernel( args.launch_info, kargs ).launch();                       \
	} break
				HYDRANT_CUDA_SHADER_IMPL_PASS( RayEmit, ray_emit );
				HYDRANT_CUDA_SHADER_IMPL_PASS( RayMarch, ray_march );
				HYDRANT_CUDA_SHADER_IMPL_PASS( Pixel, pixel );
#undef HYDRANT_CUDA_SHADER_IMPL_PASS
			}
			return ShadingResult::Ok;
		};
		return *this;
	}

	template <typename T>
	ShaderRegistrar &cpu()
	{
		_meta.devices[ ShadingDevice::Cpu ] =
		  []( void *args_ptr ) -> ShadingResult {
			auto &args = *reinterpret_cast<CpuShadingArgs *>( args_ptr );
			switch ( args.kernel_args->shading_pass ) {
#define HYDRANT_CPU_SHADER_IMPL_PASS( Pass, Lower )                                 \
	case ShadingPass::Pass: {                                                       \
		auto &kargs = *static_cast<Cpu##Pass##KernelArgs *>( args.kernel_args );    \
		kargs.launcher = (function_ptr_t)Lower##_shader_impl<typename T::Pixel, T>; \
		kargs.shader = args.shader;                                                 \
		Lower##_task_dispatch( args.thread_pool_info, kargs );                      \
	} break
				HYDRANT_CPU_SHADER_IMPL_PASS( RayEmit, ray_emit );
				HYDRANT_CPU_SHADER_IMPL_PASS( RayMarch, ray_march );
				HYDRANT_CPU_SHADER_IMPL_PASS( Pixel, pixel );
#undef HYDRANT_CPU_SHADER_IMPL_PASS
			}
			return ShadingResult::Ok;
		};
		return *this;
	}

	int build()
	{
		if ( ShaderRegistry::instance.meta.count( _type_index ) ) {
			throw std::logic_error(
			  vm::fmt( "shader '{}' already registered", _meta.class_name ) );
		}
		ShaderRegistry::instance.meta[ _type_index ] = std::move( _meta );
		return 0;
	}

private:
	std::type_index _type_index;
	ShaderMeta _meta;
};

#define REGISTER_SHADER_BUILDER( shader_builder, ... ) \
	REGISTER_SHADER_BUILDER_UNIQ_HELPER( __COUNTER__, shader_builder, __VA_ARGS__ )

#define REGISTER_SHADER_BUILDER_UNIQ_HELPER( ctr, shader_builder, ... ) \
	REGISTER_KERNEL_BUILDER_UNIQ( ctr, shader_builder, __VA_ARGS__ )

#define REGISTER_KERNEL_BUILDER_UNIQ( ctr, shader_builder, ... )                     \
	static int                                                                       \
	  shader_registrar__body__##ctr##__object =                                      \
		::hydrant::__inner__::ShaderRegistrar( typeid( __VA_ARGS__ ), #__VA_ARGS__ ) \
		  .shader_builder.build()

VM_END_MODULE()
