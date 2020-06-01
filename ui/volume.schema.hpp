#pragma once

#include <VMUtils/enum.hpp>
#include <VMUtils/json_binding.hpp>
#include <hydrant/core/glm_math.hpp>
#include <hydrant/transfer_fn.schema.hpp>

using namespace glm;
using namespace hydrant;

VM_ENUM( VolumeRenderMode,
		 Default,
		 Partition,
		 Paging );

struct VolumeRendererParams : vm::json::Serializable<VolumeRendererParams>
{
	VM_JSON_FIELD( VolumeRenderMode, mode ) = VolumeRenderMode::Default;
	VM_JSON_FIELD( TransferFnConfig, transfer_fn );
	VM_JSON_FIELD( std::size_t, mem_limit_mb ) = 1024 * 2;
};
