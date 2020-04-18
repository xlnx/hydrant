#pragma once

#include <VMUtils/enum.hpp>
#include <VMUtils/json_binding.hpp>
#include <hydrant/core/glm_math.hpp>
#include <hydrant/transfer_fn.schema.hpp>

using namespace glm;
using namespace hydrant;

struct VolumeRendererParams : vm::json::Serializable<VolumeRendererParams>
{
	VM_JSON_FIELD( TransferFnConfig, transfer_fn );
	VM_JSON_FIELD( float, density ) = 1e-2f;
	VM_JSON_FIELD( std::size_t, mem_limit_mb ) = 1024 * 2;
};
