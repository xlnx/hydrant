#pragma once

#include <VMUtils/enum.hpp>
#include <VMUtils/json_binding.hpp>
#include <hydrant/core/glm_math.hpp>

using namespace glm;
using namespace hydrant;

VM_ENUM( BlocksRenderMode,
		 Volume, Solid );

struct BlocksRendererParams : vm::json::Serializable<BlocksRendererParams>
{
	VM_JSON_FIELD( BlocksRenderMode, mode ) = BlocksRenderMode::Volume;
	VM_JSON_FIELD( float, density ) = 1e-2f;
};
