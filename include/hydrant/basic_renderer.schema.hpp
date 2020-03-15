#pragma once

#include <VMUtils/enum.hpp>
#include <VMUtils/json_binding.hpp>

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	struct BasicRendererParams : vm::json::Serializable<BasicRendererParams>
	{
		VM_JSON_FIELD( ShadingDevice, device ) = ShadingDevice::Cuda;
		VM_JSON_FIELD( std::string, device_filter ) = ".*";
		VM_JSON_FIELD( int, max_steps ) = 500;
		VM_JSON_FIELD( vec3, clear_color ) = vec3( 0 );
	};
}

VM_END_MODULE()
