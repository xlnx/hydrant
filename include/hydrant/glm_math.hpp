#pragma once

#include <glm/glm.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/io.hpp>
#include <glm/gtx/component_wise.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <VMUtils/fmt.hpp>
#include <VMUtils/attributes.hpp>
#include <VMUtils/modules.hpp>
#include <VMUtils/json_binding.hpp>

namespace glm
{
inline void to_json( nlohmann::json &j, const vec3 &v )
{
	j = { v.x, v.y, v.z };
}
inline void from_json( const nlohmann::json &j, vec3 &v )
{
	v.x = j[ 0 ].get<float>();
	v.y = j[ 1 ].get<float>();
	v.z = j[ 2 ].get<float>();
}
}  // namespace glm

VM_BEGIN_MODULE( hydrant )

using namespace glm;

VM_EXPORT
{
	struct Box3D
	{
		VM_DEFINE_ATTRIBUTE( vec3, min );
		VM_DEFINE_ATTRIBUTE( vec3, max );

		__host__ __device__ bool
		  contains( vec3 const &pt ) const
		{
			return glm::all( glm::greaterThanEqual( pt, min ) ) &&
				   glm::all( glm::lessThanEqual( pt, max ) );
		}
	};

	struct Ray
	{
		VM_DEFINE_ATTRIBUTE( vec3, o );
		VM_DEFINE_ATTRIBUTE( vec3, d );

		__host__ __device__ bool
		  intersect( Box3D const &box, float &tnear, float &tfar ) const
		{
			vec3 invr = vec3{ 1., 1., 1. } / d;
			vec3 tbot = invr * ( box.min - o );
			vec3 ttop = invr * ( box.max - o );

			vec3 tmin = min( ttop, tbot );
			vec3 tmax = max( ttop, tbot );

			tnear = glm::compMax( tmin );
			tfar = glm::compMin( tmax );

			return tfar > tnear;
		}
	};
}

VM_END_MODULE()
