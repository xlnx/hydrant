#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <VMUtils/fmt.hpp>
#include <VMUtils/attributes.hpp>
#include <VMUtils/modules.hpp>

inline std::ostream &operator<<( std::ostream &os, glm::mat4 const &m )
{
	for ( int i = 0; i != 4; ++i ) {
		vm::fprintln( os, "{} {} {} {}", m[ 0 ][ i ], m[ 1 ][ i ], m[ 2 ][ i ], m[ 3 ][ i ] );
	}
	return os;
}

inline std::ostream &operator<<( std::ostream &os, glm::vec4 const &v )
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
}

VM_END_MODULE()
