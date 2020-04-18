#pragma once

#include <VMUtils/json_binding.hpp>
#include <hydrant/core/glm_math.hpp>

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
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
}

VM_END_MODULE()
