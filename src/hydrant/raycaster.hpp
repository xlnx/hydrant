#pragma once

#include <thread>
#include <fstream>
#include <algorithm>
#include <cudafx/device.hpp>
#include <cudafx/kernel.hpp>
#include <cudafx/image.hpp>
#include <glm_math.hpp>
#include "scene.hpp"

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

struct ShaderDesc
{
	void ( *shader )() = nullptr;
	std::size_t offset = 0;

public:
	template <typename F>
	static ShaderDesc from( F const &f );

private:
	void copy_to_buffer( const void *udata, std::size_t size ) const;
};

template <typename F, bool IsRayEmitShader, bool IsPixelShader>
struct ShaderRegistererImpl;

/* Ray Emit Shader Impl */

template <typename P, typename F>
static __device__ void
  ray_emit_shader_impl( Ray const &ray, P *pixel, F const *shader )
{
	return shader->apply( ray, *pixel );
}

template <typename P, typename F>
using p_ray_emit_shader_t = void ( * )( Ray const &, P *, F const * );

template <typename P, typename F>
__device__ p_ray_emit_shader_t<P, F> p_ray_emit_shader = ray_emit_shader_impl<P, F>;

template <typename F>
struct ShaderRegistererImpl<F, true, false>
{
	static void apply( ShaderDesc &args, F const &f )
	{
		cudaMemcpyFromSymbol( &args.shader, p_ray_emit_shader<typename F::Pixel, F>, sizeof( args.shader ) );
	}
};

struct RayEmitShaderArguments
{
	ViewArguments view;
	ImageDesc img;
	ShaderDesc shader;
};

extern cufx::Kernel<void( RayEmitShaderArguments args )> ray_emit_kernel;

VM_EXPORT
{
	struct RayEmitShader
	{
	};
}

/* Pixel Shader Impl */

template <typename P, typename F>
static __device__ void
  pixel_shader_impl( P *pixel, F const *shader )
{
	return shader->apply( ray, *pixel );
}

template <typename P, typename F>
using p_pixel_shader_t = void ( * )( P *, F const * );

template <typename P, typename F>
__device__ p_pixel_shader_t<P, F> p_pixel_shader = pixel_shader_impl<P, F>;

template <typename F>
struct ShaderRegistererImpl<F, false, true>
{
	static void apply( ShaderDesc &args, F const &f )
	{
		cudaMemcpyFromSymbol( &args.shader, p_pixel_shader<typename F::Pixel, F>, sizeof( args.shader ) );
	}
};

struct PixelShaderArguments
{
	ImageDesc img;
	ShaderDesc shader;
};

extern cufx::Kernel<void( PixelShaderArguments args )> pixel_kernel;

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

/* Raycaster */

VM_EXPORT
{
#define SHADER_DECL( F ) \
	extern template class F

#define SHADER_IMPL( F )                             \
	template <>                                      \
	static ShaderDesc ShaderDesc::from( F const &f ) \
	{                                                \
		ShaderDesc desc;                             \
		ShaderRegisterer::apply( desc, f );          \
		copy_to_buffer( &f, sizeof( f ) );           \
		return desc;                                 \
	}                                                \
	template class F

	struct Raycaster
	{
		template <typename P, typename F>
		void cast( Exhibit const &e, Camera const &c, cufx::ImageView<P> &img, F const &f )
		{
			auto args = get_ray_emitter_args( e, c, img );
			CastImpl<P, F, true>::apply( args, img, f, this );
		}
		template <typename P, typename F>
		void cast( cufx::ImageView<P> &img )
		{
			auto args = get_ray_emitter_args( e, c, img );
			CastImpl<P, F, true>::apply( args, img, f, this );
		}

	private:
		template <typename P, typename F, bool IsGpuFn>
		struct CastImpl;

		template <typename P, typename F, bool IsGpuFn>
		friend struct CastImpl;

		template <typename P>
		RayEmitShaderArguments get_ray_emitter_args( Exhibit const &e, Camera const &c, cufx::ImageView<P> &img )
		{
			auto d = max( abs( e.center - e.size ), abs( e.center ) );
			float scale = glm::compMax( d );
			mat4 et = { { scale, 0, 0, 0 },
						{ 0, scale, 0, 0 },
						{ 0, 0, scale, 0 },
						{ e.center.x, e.center.y, e.center.z, 1 } };
			auto ivt = inverse( c.get_matrix() );

			RayEmitShaderArguments cast_opts;
			cast_opts.trans = et * ivt;
			cast_opts.itg_fovy = 1. / tan( M_PI / 3 / 2 );
			cast_opts.ray_o = et * vec4{ c.position.x, c.position.y, c.position.z, 1 };
			cast_opts.resolution = ivec2{ img.width(), img.height() };
			cast_opts.pixel_size = sizeof( P );

			return cast_opts;
		}

		template <typename P, typename F>
		void cpu_cast_impl( RayEmitShaderArguments const &args, cufx::ImageView<P> &img, F const &f )
		{
			auto cc = vec2( args.resolution ) / 2.f;
			auto nthreads = std::thread::hardware_concurrency();
			std::vector<std::thread> threads;
			for ( int i = 0; i != nthreads; ++i ) {
				threads.emplace_back(
				  [&, y0 = i] {
					  Ray ray = { args.ray_o, {} };
					  for ( int y = y0; y < args.resolution.y; y += nthreads ) {
						  for ( int x = 0; x < args.resolution.x; ++x ) {
							  auto uv = ( vec2{ x, y } - cc ) * 2.f / float( args.resolution.y );
							  ray.d = normalize( vec3( args.trans * vec4( uv.x, -uv.y, -args.itg_fovy, 1 ) ) - ray.o );
							  f( ray, img.at( x, y ) );
						  }
					  }
				  } );
			}
			for ( auto &t : threads ) { t.join(); }
		}
	};

	template <typename P, typename F>
	struct Raycaster::CastImpl<P, F, false>
	{
		static void apply( RayEmitShaderArguments &args, cufx::ImageView<P> &img, F const &f, Raycaster *self )
		{
			args.image = reinterpret_cast<char *>( &img.at_host( 0, 0 ) );
			self->cpu_cast_impl( args, img, f );
		}
	};

	template <typename P, typename F>
	struct Raycaster::CastImpl<P, F, true>
	{
		static int round_up_div( int a, int b )
		{
			return a % b == 0 ? a / b : a / b + 1;
		}
		static void apply( RayEmitShaderArguments &args, cufx::ImageView<P> &img, F const &f, Raycaster *self )
		{
			static_assert( std::is_trivially_copyable<F>::value, "shader must be trivially copyable" );

			args.image.data = reinterpret_cast<char *>( &img.at_device( 0, 0 ) );
			args.shader = ShaderDesc::from( f );

			auto kernel_block_dim = dim3( 32, 32 );
			auto launch_info = cufx::KernelLaunchInfo{}
								 .set_device( cufx::Device::scan()[ 0 ] )
								 .set_grid_dim( round_up_div( args.resolution.x, kernel_block_dim.x ),
												round_up_div( args.resolution.y, kernel_block_dim.y ) )
								 .set_block_dim( kernel_block_dim );
			ray_emit_kernel( launch_info, args ).launch();
		}
	};
}

VM_END_MODULE()
