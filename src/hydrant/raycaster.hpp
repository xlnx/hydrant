#pragma once

#include <thread>
#include <fstream>
#include <algorithm>
#include <cudafx/device.hpp>
#include <cudafx/kernel.hpp>
#include <cudafx/image.hpp>
#include <glm_math.hpp>

VM_BEGIN_MODULE( hydrant )

using namespace glm;

struct PTU : vm::json::Serializable<PTU>
{
	VM_JSON_FIELD( vec3, position );
	VM_JSON_FIELD( vec3, target ) = { 0, 0, 0 };
	VM_JSON_FIELD( vec3, up ) = { 0, 1, 0 };
};

struct Orbit : vm::json::Serializable<Orbit>
{
	VM_JSON_FIELD( vec3, center ) = { 0, 0, 0 };
	VM_JSON_FIELD( vec3, arm );
};

struct CameraConfig : vm::json::Serializable<CameraConfig>
{
	VM_JSON_FIELD( std::shared_ptr<PTU>, ptu ) = nullptr;
	VM_JSON_FIELD( std::shared_ptr<Orbit>, orbit ) = nullptr;
};

using shader_t = void ( * )( Ray const &, void *, void const * );

template <typename P, typename F>
using typed_shader_t = void ( * )( Ray const &, P *, F const * );

struct CastOptions
{
	mat4 trans;
	float itg_fovy;
	vec3 ray_o;
	ivec2 resolution;
	size_t pixel_size;
	char *image;
	shader_t shader = nullptr;
	std::size_t udata_offset = 0;
};

extern cufx::Kernel<void( CastOptions opts )> cast_kernel;

extern void copy_to_argument_buffer( std::size_t offset, const void *udata, std::size_t size );

template <typename P, typename F>
static __device__ void
  apply_impl( Ray const &ray, P *pixel, F const *shader )
{
	return shader->apply( ray, *pixel );
}

template <typename P, typename F>
__device__ typed_shader_t<P, F> p_apply = apply_impl<P, F>;

VM_EXPORT
{
	struct Exhibit : vm::Dynamic
	{
		VM_DEFINE_ATTRIBUTE( vec3, size );
		VM_DEFINE_ATTRIBUTE( vec3, center );
	};

	struct Camera
	{
		VM_DEFINE_ATTRIBUTE( vec3, position ) = { 2, 0.5, 0 };
		VM_DEFINE_ATTRIBUTE( vec3, target ) = { 0, 0, 0 };
		VM_DEFINE_ATTRIBUTE( vec3, up ) = { 0, 1, 0 };

		mat4 get_matrix() const { return lookAt( position, target, up ); }

	public:
		static Camera from_config( std::string const &filename )
		{
			CameraConfig cfg;
			std::ifstream is( filename );
			is >> cfg;

			Camera camera;
			if ( cfg.ptu ) {
				camera.position = cfg.ptu->position;
				camera.target = cfg.ptu->target;
				camera.up = cfg.ptu->up;
			} else if ( cfg.orbit ) {
				camera.target = cfg.orbit->center;
				mat4 m = { { 1, 0, 0, 0 },
						   { 0, 1, 0, 0 },
						   { 0, 0, 1, 0 },
						   { 0, 0, 0, 1 } };
				m = rotate( m, radians( cfg.orbit->arm.y ), vec3{ 0, 0, 1 } );
				m = rotate( m, radians( cfg.orbit->arm.x ), vec3{ 0, 1, 0 } );
				camera.position = m * vec4{ cfg.orbit->arm.z, 0, 0, 1 };
			}
			return camera;
		}
	};

	struct Raycaster;

	template <typename F>
	struct ShaderRegisterer
	{
	private:
		static void fill_opts( CastOptions &opts, cufx::ImageView<typename F::Pixel> &img, F const &f );

		friend struct Raycaster;
	};

#define SHADER_DECL( F ) \
	extern template class F

#define SHADER_IMPL( F )                                                                                          \
	template <>                                                                                                   \
	void ShaderRegisterer<F>::fill_opts( CastOptions &opts, cufx::ImageView<typename F::Pixel> &img, F const &f ) \
	{                                                                                                             \
		opts.image = reinterpret_cast<char *>( &img.at_device( 0, 0 ) );                                          \
		cudaMemcpyFromSymbol( &opts.shader, p_apply<typename F::Pixel, F>, sizeof( opts.shader ) );               \
		opts.udata_offset = 0;                                                                                    \
		copy_to_argument_buffer( opts.udata_offset, &f, sizeof( F ) );                                            \
	}                                                                                                             \
	template class F

	struct Raycaster
	{
		template <typename P, typename F>
		void cast( Exhibit const &e, Camera const &c, cufx::ImageView<P> &img, F const &f )
		{
			auto opts = get_basic_cast_opts( e, c, img );
			CastImpl<P, F, !std::is_function<F>::value>::apply( opts, img, f, this );
		}

	private:
		template <typename P, typename F, bool IsGpuFn>
		struct CastImpl;

		template <typename P, typename F, bool IsGpuFn>
		friend struct CastImpl;

		template <typename P>
		CastOptions get_basic_cast_opts( Exhibit const &e, Camera const &c, cufx::ImageView<P> &img )
		{
			auto d = max( abs( e.center - e.size ), abs( e.center ) );
			float scale = glm::compMax( d );
			mat4 et = { { scale, 0, 0, 0 },
						{ 0, scale, 0, 0 },
						{ 0, 0, scale, 0 },
						{ e.center.x, e.center.y, e.center.z, 1 } };
			auto ivt = inverse( c.get_matrix() );

			CastOptions cast_opts;
			cast_opts.trans = et * ivt;
			cast_opts.itg_fovy = 1. / tan( M_PI / 3 / 2 );
			cast_opts.ray_o = et * vec4{ c.position.x, c.position.y, c.position.z, 1 };
			cast_opts.resolution = ivec2{ img.width(), img.height() };
			cast_opts.pixel_size = sizeof( P );

			return cast_opts;
		}

		template <typename P, typename F>
		void cpu_cast_impl( CastOptions const &opts, cufx::ImageView<P> &img, F const &f )
		{
			auto cc = vec2( opts.resolution ) / 2.f;
			auto nthreads = std::thread::hardware_concurrency();
			std::vector<std::thread> threads;
			for ( int i = 0; i != nthreads; ++i ) {
				threads.emplace_back(
				  [&, y0 = i] {
					  Ray ray = { opts.ray_o, {} };
					  for ( int y = y0; y < opts.resolution.y; y += nthreads ) {
						  for ( int x = 0; x < opts.resolution.x; ++x ) {
							  auto uv = ( vec2{ x, y } - cc ) * 2.f / float( opts.resolution.y );
							  ray.d = normalize( vec3( opts.trans * vec4( uv.x, -uv.y, -opts.itg_fovy, 1 ) ) - ray.o );
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
		static void apply( CastOptions &opts, cufx::ImageView<P> &img, F const &f, Raycaster *self )
		{
			opts.image = reinterpret_cast<char *>( &img.at_host( 0, 0 ) );
			self->cpu_cast_impl( opts, img, f );
		}
	};

	template <typename P, typename F>
	struct Raycaster::CastImpl<P, F, true>
	{
		static int round_up_div( int a, int b )
		{
			return a % b == 0 ? a / b : a / b + 1;
		}
		static void apply( CastOptions &opts, cufx::ImageView<P> &img, F const &f, Raycaster *self )
		{
			static_assert( std::is_trivially_copyable<F>::value, "shader must be trivially copyable" );

			ShaderRegisterer<F>::fill_opts( opts, img, f );

			auto kernel_block_dim = dim3( 32, 32 );
			auto launch_info = cufx::KernelLaunchInfo{}
								 .set_device( cufx::Device::scan()[ 0 ] )
								 .set_grid_dim( round_up_div( opts.resolution.x, kernel_block_dim.x ),
												round_up_div( opts.resolution.y, kernel_block_dim.y ) )
								 .set_block_dim( kernel_block_dim );
			cast_kernel( launch_info, opts ).launch();
		}
	};
}

VM_END_MODULE()
