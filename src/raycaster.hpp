#pragma once

#include <algorithm>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <VMUtils/fmt.hpp>
#include <VMUtils/attributes.hpp>
#include <cudafx/image.hpp>

std::ostream &operator<<( std::ostream &os, glm::mat4 const &m )
{
	for ( int i = 0; i != 4; ++i ) {
		vm::fprintln( os, "{} {} {} {}", m[ 0 ][ i ], m[ 1 ][ i ], m[ 2 ][ i ], m[ 3 ][ i ] );
	}
	return os;
}

std::ostream &operator<<( std::ostream &os, glm::vec4 const &v )
{
	for ( int i = 0; i != 4; ++i ) {
		vm::fprintln( os, "{}", v[ i ] );
	}
	return os;
}

VM_BEGIN_MODULE( hydrant )

using namespace glm;

VM_EXPORT
{
	struct Box3D
	{
		VM_DEFINE_ATTRIBUTE( vec3, min );
		VM_DEFINE_ATTRIBUTE( vec3, max );
	};

	struct Ray
	{
		VM_DEFINE_ATTRIBUTE( vec3, o );
		VM_DEFINE_ATTRIBUTE( vec3, d );

		bool intersect( Box3D const &box, float &tnear, float &tfar ) const
		{
			vec3 invr = vec3{ 1., 1., 1. } / d;
			vec3 tbot = invr * ( box.min - o );
			vec3 ttop = invr * ( box.max - o );

			vec3 tmin = min( ttop, tbot );
			vec3 tmax = max( ttop, tbot );

			tnear = max( max( tmin.x, tmin.y ), tmin.z );
			tfar = min( min( tmax.x, tmax.y ), tmax.z );

			return tfar > tnear;
		}
	};

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
			// vm::println( "{}", et );
			// vm::println( "{}", ivt );
			// vm::println( "{}", et * ivt * vec4{ 1, 0, -2.2, 1 } );

			auto itg_fovy = 1. / tan( M_PI / 3 / 2 );
			auto tt = et * ivt;
			auto cc = vec2{ img.get_width(), img.get_height() } / 2.f;

			Ray ray = { et * vec4{ c.position.x, c.position.y, c.position.z, 1 }, {} };

			for ( int y = 0; y != img.get_height(); ++y ) {
				for ( int x = 0; x != img.get_width(); ++x ) {
					auto uv = ( vec2{ x, y } - cc ) * 2.f / float( img.get_height() );
					ray.d = vec3( tt * vec4( uv.x, uv.y, -itg_fovy, 1 ) ) - ray.o;
					// vm::println( "{}", std::make_tuple( vv.x, vv.y, vv.z, vv.w ) );
					img.at( x, y ) = f( ray );
				}
			}
		}
	};
}

VM_END_MODULE()
