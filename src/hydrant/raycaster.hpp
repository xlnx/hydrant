#pragma once

#include <thread>
#include <fstream>
#include <algorithm>
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

struct CastOptions
{
	mat4 trans;
	float itg_fovy;
	vec3 ray_o;
	ivec2 resolution;
	char *image;
	size_t pixel_size;
	void ( *shader )( Ray const &, void * );
};

extern cufx::Kernel<void( CastOptions opts )> cast_kernel;

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

	struct Raycaster
	{
		template <typename P, typename F>
		void cast( Exhibit const &e, Camera const &c, cufx::Image<P> &img, F const &f )
		{
			cast_impl( get_basic_cast_opts( e, c, img ), img, f );
		}

		template <typename P>
		void cast( Exhibit const &e, Camera const &c, cufx::Image<P> &img, P ( *f )( Ray const & ) )
		{
			auto cast_opts = get_basic_cast_opts( e, c, img );
			auto launch_info = cufx::KernelLaunchInfo{};
			cast_kernel( launch_info, cast_opts ).launch();
		}

	private:
		template <typename P>
		CastOptions get_basic_cast_opts( Exhibit const &e, Camera const &c, cufx::Image<P> &img )
		{
			auto d = max( abs( e.center - e.size ), abs( e.center ) );
			float scale = max( d.x, max( d.y, d.z ) );
			mat4 et = { { scale, 0, 0, 0 },
						{ 0, scale, 0, 0 },
						{ 0, 0, scale, 0 },
						{ e.center.x, e.center.y, e.center.z, 1 } };
			auto ivt = inverse( c.get_matrix() );

			CastOptions cast_opts;
			cast_opts.trans = et * ivt;
			cast_opts.itg_fovy = 1. / tan( M_PI / 3 / 2 );
			cast_opts.ray_o = et * vec4{ c.position.x, c.position.y, c.position.z, 1 };
			cast_opts.resolution = ivec2{ img.get_width(), img.get_height() };
			cast_opts.image = reinterpret_cast<char *>( &img.at( 0, 0 ) );
			cast_opts.pixel_size = sizeof( P );

			return cast_opts;
		}

		template <typename P, typename F>
		void cast_impl( CastOptions const &opts, cufx::Image<P> &img, F const &f )
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
							  img.at( x, y ) = f( ray );
						  }
					  }
				  } );
			}
			for ( auto &t : threads ) { t.join(); }
		}
	};
}

VM_END_MODULE()
