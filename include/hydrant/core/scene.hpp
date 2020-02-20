#pragma once

#include <fstream>
#include <VMUtils/modules.hpp>
#include <VMUtils/concepts.hpp>
#include <VMUtils/attributes.hpp>
#include <hydrant/core/glm_math.hpp>

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

	struct CameraPtu : vm::json::Serializable<CameraPtu>
	{
		VM_JSON_FIELD( vec3, position );
		VM_JSON_FIELD( vec3, target ) = { 0, 0, 0 };
		VM_JSON_FIELD( vec3, up ) = { 0, 1, 0 };
	};

	struct CameraOrbit : vm::json::Serializable<CameraOrbit>
	{
		VM_JSON_FIELD( vec3, center ) = { 0, 0, 0 };
		VM_JSON_FIELD( vec3, arm );
	};

	struct CameraPerspective : vm::json::Serializable<CameraPerspective>
	{
		VM_JSON_FIELD( float, fovy ) = 60;
	};

	struct CameraConfig : vm::json::Serializable<CameraConfig>
	{
		VM_JSON_FIELD( std::shared_ptr<CameraPtu>, ptu ) = nullptr;
		VM_JSON_FIELD( std::shared_ptr<CameraOrbit>, orbit ) = nullptr;
		VM_JSON_FIELD( std::shared_ptr<CameraPerspective>, perspective ) =
		  std::make_shared<CameraPerspective>( CameraPerspective{}
												 .set_fovy( 60 ) );
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
