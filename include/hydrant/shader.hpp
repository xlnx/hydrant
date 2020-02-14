#pragma once

#include <algorithm>
#include <cudafx/kernel.hpp>
#include <cudafx/image.hpp>
#include <hydrant/glm_math.hpp>

VM_BEGIN_MODULE( hydrant )

struct ViewArguments
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
};

template <typename P>
ImageDesc make_image_desc( cufx::ImageView<P> const &img )
{
	ImageDesc desc;
	desc.resolution = ivec2{ img.width(), img.height() };
	desc.pixel_size = sizeof( P );
	desc.data = reinterpret_cast<char *>( &img.at_device( 0, 0 ) );
	return desc;
}

struct ShaderDesc
{
	void ( *shader )() = nullptr;
	std::size_t offset = 0;

	void copy_to_buffer( const void *udata, std::size_t size ) const;
};

VM_EXPORT
{
	template <typename F>
	ShaderDesc make_shader_desc( F const &f, std::size_t offset );
}

template <typename F, bool IsRayEmitShader, bool IsPixelShader>
struct ShaderRegistererImpl
{
	static_assert( IsRayEmitShader ^ IsPixelShader, "invalid shader config: " );
};

/* Ray Emit Shader Impl */

template <typename P, typename F>
static __device__ void
  ray_emit_shader_impl( Ray const &ray, void *pixel, void const *shader )
{
	return reinterpret_cast<F const *>( shader )->apply(
	  ray, *reinterpret_cast<P *>( pixel ) );
}

using p_ray_emit_shader_t = void ( * )( Ray const &, void *, void const * );

template <typename P, typename F>
__device__ p_ray_emit_shader_t p_ray_emit_shader = ray_emit_shader_impl<P, F>;

template <typename F>
struct ShaderRegistererImpl<F, true, false>
{
	static void apply( ShaderDesc &args, F const &f )
	{
		cudaMemcpyFromSymbol( &args.shader, p_ray_emit_shader<typename F::Pixel, F>, sizeof( args.shader ) );
	}
};

struct RayEmitKernelArgs
{
	ViewArguments view;
	ImageDesc image;
	ShaderDesc shader;
};

extern cufx::Kernel<void( RayEmitKernelArgs args )> ray_emit_kernel;

VM_EXPORT
{
	struct RayEmitShader
	{
	};
}

/* Pixel Shader Impl */

template <typename P, typename F>
static __device__ void
  pixel_shader_impl( void *pixel, void const *shader )
{
	return reinterpret_cast<F const *>( shader )->apply(
	  *reinterpret_cast<P *>( pixel ) );
}

using p_pixel_shader_t = void ( * )( void *, void const * );

template <typename P, typename F>
__device__ p_pixel_shader_t p_pixel_shader = pixel_shader_impl<P, F>;

template <typename F>
struct ShaderRegistererImpl<F, false, true>
{
	static void apply( ShaderDesc &args, F const &f )
	{
		cudaMemcpyFromSymbol( &args.shader, p_pixel_shader<typename F::Pixel, F>, sizeof( args.shader ) );
	}
};

struct PixelKernelArgs
{
	ImageDesc image;
	ShaderDesc shader;
};

extern cufx::Kernel<void( PixelKernelArgs args )> pixel_kernel;

VM_EXPORT
{
	struct PixelShader
	{
	};
}

/* Shader Registerer */

template <typename F>
struct ShaderRegisterer : ShaderRegistererImpl<
							F,
							std::is_base_of<RayEmitShader, F>::value,
							std::is_base_of<PixelShader, F>::value>
{
};

#define SHADER_IMPL( F )                                             \
	template <>                                                      \
	ShaderDesc make_shader_desc<F>( F const &f, std::size_t offset ) \
	{                                                                \
		ShaderDesc desc;                                             \
		desc.offset = offset;                                        \
		ShaderRegisterer<F>::apply( desc, f );                       \
		desc.copy_to_buffer( &f, sizeof( f ) );                      \
		return desc;                                                 \
	}

VM_END_MODULE()
