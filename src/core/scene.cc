#include <glog/logging.h>
#include <hydrant/core/scene.hpp>

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	Camera::Camera( CameraConfig const &cfg )
	{
		if ( cfg.ptu ) {
			update_params( *cfg.ptu );
		} else if ( cfg.orbit ) {
			update_params( *cfg.orbit );
		} else {
			LOG( FATAL ) << "invalid config";
		}
		if ( cfg.perspective ) {
			ctg_fovy_2 = 1.f / tan( radians( cfg.perspective->fovy / 2.f ) );
		}
	}

	Camera &Camera::update_params( CameraPtu const &ptu )
	{
		position = ptu.position;
		target = ptu.target;
		up = ptu.up;
		return *this;
	}

	Camera &Camera::update_params( CameraOrbit const &orbit )
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
}

VM_END_MODULE()
