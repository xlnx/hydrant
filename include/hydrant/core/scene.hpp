#pragma once

#include <fstream>
#include <VMUtils/modules.hpp>
#include <VMUtils/concepts.hpp>
#include <VMUtils/attributes.hpp>
#include <hydrant/core/glm_math.hpp>

VM_BEGIN_MODULE( hydrant )

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

VM_EXPORT
{
	struct Exhibit : vm::Dynamic
	{
		VM_DEFINE_ATTRIBUTE( vec3, size );
		VM_DEFINE_ATTRIBUTE( vec3, center );

		mat4 get_matrix() const
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

	struct CameraConfig : vm::json::Serializable<CameraConfig>
	{
		VM_JSON_FIELD( std::shared_ptr<PTU>, ptu ) = nullptr;
		VM_JSON_FIELD( std::shared_ptr<Orbit>, orbit ) = nullptr;
	};

	struct Camera
	{
		VM_DEFINE_ATTRIBUTE( vec3, position ) = { 2, 0.5, 0 };
		VM_DEFINE_ATTRIBUTE( vec3, target ) = { 0, 0, 0 };
		VM_DEFINE_ATTRIBUTE( vec3, up ) = { 0, 1, 0 };

		mat4 get_matrix() const { return lookAt( position, target, up ); }

	public:
		static Camera from_config( CameraConfig const &cfg )
		{
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
}

VM_END_MODULE()
