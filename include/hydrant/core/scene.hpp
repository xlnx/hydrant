#pragma once

#include <fstream>
#include <VMUtils/modules.hpp>
#include <VMUtils/concepts.hpp>
#include <VMUtils/attributes.hpp>
#include <hydrant/core/glm_math.hpp>
#include <hydrant/core/scene.schema.hpp>

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	struct Exhibit : vm::Dynamic
	{
		VM_DEFINE_ATTRIBUTE( vec3, size );
		VM_DEFINE_ATTRIBUTE( vec3, center );

		mat4 get_iet() const
		{
			auto d = max( abs( center - size ), abs( center ) );
			float scale = glm::compMax( d );
			mat4 et = { { scale, 0, 0, 0 },
						{ 0, scale, 0, 0 },
						{ 0, 0, scale, 0 },
						{ center.x, center.y, center.z, 1 } };
			return et;
		}
	};

	struct Camera
	{
		VM_DEFINE_ATTRIBUTE( vec3, position ) = { 2, 0.5, 0 };
		VM_DEFINE_ATTRIBUTE( vec3, target ) = { 0, 0, 0 };
		VM_DEFINE_ATTRIBUTE( vec3, up ) = { 0, 1, 0 };

		VM_DEFINE_ATTRIBUTE( float, ctg_fovy_2 ) = INFINITY;

	public:
		Camera() = default;

		Camera( CameraConfig const &cfg );

	public:
		mat4 get_ivt() const { return inverse( lookAt( position, target, up ) ); }

		Camera &update_params( CameraPtu const &ptu );

		Camera &update_params( CameraOrbit const &orbit );
	};
}

VM_END_MODULE()
