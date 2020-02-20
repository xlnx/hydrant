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

	struct CameraConfig : vm::json::Serializable<CameraConfig>
	{
		VM_JSON_FIELD( std::shared_ptr<CameraPtu>, ptu ) = nullptr;
		VM_JSON_FIELD( std::shared_ptr<CameraOrbit>, orbit ) = nullptr;
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
			if ( cfg.ptu ) {
				return from_ptu( *cfg.ptu );
			} else if ( cfg.orbit ) {
				return from_orbit( *cfg.orbit );
			}
			throw std::logic_error( "invalid config" );
		}
		static Camera from_ptu( CameraPtu const &ptu )
		{
			Camera camera;
			camera.position = ptu.position;
			camera.target = ptu.target;
			camera.up = ptu.up;
			return camera;
		}
		static Camera from_orbit( CameraOrbit const &orbit )
		{
			Camera camera;
			camera.target = orbit.center;
			// mat4 m = { { 1, 0, 0, 0 },
			// 		   { 0, 1, 0, 0 },
			// 		   { 0, 0, 1, 0 },
			// 		   { 0, 0, 0, 1 } };
			// m = ;
			// m = rotate( m, radians( orbit.arm.x ), vec3{ 0, 1, 0 } );
			camera.position = rotate( mat4( 1 ),
									  radians( orbit.arm.y ),
									  vec3( 0, 0, 1 ) ) *
							  vec4( orbit.arm.z, 0, 0, 1 );
			camera.position = rotate( mat4( 1 ),
									  radians( orbit.arm.x ),
									  vec3( 0, 1, 0 ) ) *
							  vec4( camera.position, 1 );
			return camera;
		}
	};
}

VM_END_MODULE()
