#pragma once

#include <thread>
#include <algorithm>
#include <cudafx/image.hpp>
#include <VMUtils/json_binding.hpp>
#include "math.hpp"

namespace glm
{
void to_json( nlohmann::json &j, const vec3 &v )
{
	j = { v.x, v.y, v.z };
}
void from_json( const nlohmann::json &j, vec3 &v )
{
	v.x = j[ 0 ].get<float>();
	v.y = j[ 1 ].get<float>();
	v.z = j[ 2 ].get<float>();
}
}  // namespace glm

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
				auto m = identity<mat4>();
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
			auto d = max( abs( e.center - e.size ), abs( e.center ) );
			float scale = max( d.x, max( d.y, d.z ) );
			vec3 translate = { e.center.x, e.center.y, e.center.z };
			mat4 et = { { scale, 0, 0, 0 },
						{ 0, scale, 0, 0 },
						{ 0, 0, scale, 0 },
						{ e.center.x, e.center.y, e.center.z, 1 } };
			auto ivt = inverse( c.get_matrix() );

			auto itg_fovy = 1. / tan( M_PI / 3 / 2 );
			auto tt = et * ivt;
			auto cc = vec2{ img.get_width(), img.get_height() } / 2.f;

			auto nthreads = std::thread::hardware_concurrency();
			std::vector<std::thread> threads;
			for ( int i = 0; i != nthreads; ++i ) {
				threads.emplace_back(
				  [&, y0 = i] {
					  Ray ray = { et * vec4{ c.position.x, c.position.y, c.position.z, 1 }, {} };
					  for ( int y = y0; y < img.get_height(); y += nthreads ) {
						  for ( int x = 0; x < img.get_width(); ++x ) {
							  auto uv = ( vec2{ x, y } - cc ) * 2.f / float( img.get_height() );
							  ray.d = normalize( vec3( tt * vec4( uv.x, -uv.y, -itg_fovy, 1 ) ) - ray.o );
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
