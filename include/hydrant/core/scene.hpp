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

		Camera( CameraConfig const &cfg )
		{
			if ( cfg.ptu ) {
				update_params( *cfg.ptu );
			} else if ( cfg.orbit ) {
				update_params( *cfg.orbit );
			} else {
				throw std::logic_error( "invalid config" );
			}
			if ( cfg.perspective ) {
				ctg_fovy_2 = 1.f / tan( radians( cfg.perspective->fovy / 2.f ) );
			}
		}

	public:
		mat4 get_matrix() const { return lookAt( position, target, up ); }

		Camera &update_params( CameraPtu const &ptu )
		{
			position = ptu.position;
			target = ptu.target;
			up = ptu.up;
			return *this;
		}

		Camera &update_params( CameraOrbit const &orbit )
		{
			target = orbit.center;
			position = rotate( mat4( 1 ),
							   radians( orbit.arm.y ),
							   vec3( 0, 0, 1 ) ) *
					   vec4( orbit.arm.z, 0, 0, 1 );
			position = rotate( mat4( 1 ),
							   radians( orbit.arm.x ),
							   vec3( 0, 1, 0 ) ) *
					   vec4( position, 1 );
			up = vec3( 0, 1, 0 );
			return *this;
		}
	};
}

VM_END_MODULE()
