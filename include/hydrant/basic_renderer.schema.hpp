#pragma once

#include <VMUtils/enum.hpp>
#include <VMUtils/json_binding.hpp>
#include <hydrant/core/shader.schema.hpp>
#include <hydrant/core/glm_math.hpp>

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	struct BasicRendererParams : vm::json::Serializable<BasicRendererParams>
	{
		VM_JSON_FIELD( ShadingDevice, device ) = ShadingDevice::Cuda;
		VM_JSON_FIELD( int, comm_rank ) = 0;
		VM_JSON_FIELD( float, sample_rate ) = 1.0;
		VM_JSON_FIELD( int, max_steps ) = 4000000;
		VM_JSON_FIELD( vec3, clear_color ) = vec3( 0 );
	};
}

VM_END_MODULE()
