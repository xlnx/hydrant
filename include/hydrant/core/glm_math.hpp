#pragma once

#include <cuda_runtime.h>
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
template <length_t N, typename T, qualifier Q>
void to_json( nlohmann::json &j, const vec<N, T, Q> &v )
{
	j = nlohmann::json::array();
	for ( auto i = 0; i < N; ++i ) {
		j[ i ] = v[ i ];
	}
}

template <length_t N, typename T, qualifier Q>
inline void from_json( const nlohmann::json &j, vec<N, T, Q> &v )
{
	for ( auto i = 0; i < N; ++i ) {
		v[ i ] = j[ i ].get<T>();
	}
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
