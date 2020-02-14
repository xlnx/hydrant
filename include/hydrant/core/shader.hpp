#pragma once

#include <map>
#include <string>
#include <functional>
#include <typeindex>
#include <typeinfo>
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
	enum ShadingDevice
	{
		Cpu,
		Cuda
	};

	enum ShadingResult
	{
		Ok = 0,
		Err
	};

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
	Pixel
};

struct ViewArgs
{
	mat4 trans;
	float itg_fovy;
	vec3 ray_o;
};

struct ImageDesc
{
	ivec2 resolution;
	size_t pixel_size;
	char *data;

	template <typename P>
	void create_from_img( cufx::ImageView<P> const &img )
	{
		resolution = ivec2{ img.width(), img.height() };
		pixel_size = sizeof( P );
		data = reinterpret_cast<char *>( &img.at_device( 0, 0 ) );
	}
};

struct DeviceFunctionDesc
{
	void ( *fp )() = nullptr;
	std::size_t offset = 0;

	void copy_to_buffer( const void *udata, std::size_t size ) const;
};

struct BasicCudaShadingKernelArgs
{
	ShadingPass shading_pass;
	ImageDesc image_desc;
	DeviceFunctionDesc function_desc;
};

struct CudaRayEmitShadingKernelArgs : BasicCudaShadingKernelArgs
{
	ViewArgs view;
};

struct CudaPixelShadingKernelArgs : BasicCudaShadingKernelArgs
{
};

struct CudaShadingArgs
{
	cufx::KernelLaunchInfo launch_info;
	BasicCudaShadingKernelArgs *kernel_args;
	IShaderTypeErased const *shader;
};

struct ShaderMeta
{
	std::map<ShadingDevice,
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

template <typename P, typename F>
__device__ void
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

using p_ray_emit_shader_t = void ( * )( Ray const &, void *, void const * );

template <typename P, typename F>
__device__ p_ray_emit_shader_t p_ray_emit_shader = ray_emit_shader_impl<P, F>;

template <typename P, typename F>
__device__ void
  pixel_shader_impl( void *pixel_in_out, void const *shader_in )
{
	auto &pixel = *reinterpret_cast<P *>( pixel_in_out );
	F const &shader = *reinterpret_cast<F const *>( shader_in );
	auto pixel_reg = pixel;

	shader.main( pixel_reg );

	pixel = pixel_reg;
}

using p_pixel_shader_t = void ( * )( void *, void const * );

template <typename P, typename F>
__device__ p_pixel_shader_t p_pixel_shader = pixel_shader_impl<P, F>;

extern cufx::Kernel<void( CudaRayEmitShadingKernelArgs args )> ray_emit_kernel;
extern cufx::Kernel<void( CudaPixelShadingKernelArgs args )> pixel_kernel;

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
			if ( args.kernel_args->shading_pass == ShadingPass::RayEmit ) {
				auto &ray_emit_args = *static_cast<CudaRayEmitShadingKernelArgs *>( args.kernel_args );
				cudaMemcpyFromSymbol( &ray_emit_args.function_desc.fp,
									  p_ray_emit_shader<typename T::Pixel, T>,
									  sizeof( ray_emit_args.function_desc.fp ) );
				ray_emit_args.function_desc.offset = 0;
				ray_emit_args.function_desc.copy_to_buffer( args.shader, sizeof( T ) );
				ray_emit_kernel( args.launch_info, ray_emit_args ).launch();
			} else {
				auto &pixel_args = *static_cast<CudaPixelShadingKernelArgs *>( args.kernel_args );
				cudaMemcpyFromSymbol( &pixel_args.function_desc.fp,
									  p_pixel_shader<typename T::Pixel, T>,
									  sizeof( pixel_args.function_desc.fp ) );
				pixel_args.function_desc.offset = 0;
				pixel_args.function_desc.copy_to_buffer( args.shader, sizeof( T ) );
				pixel_kernel( args.launch_info, pixel_args ).launch();
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
