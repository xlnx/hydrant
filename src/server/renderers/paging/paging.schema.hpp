#pragma once

#include <hydrant/core/glm_math.hpp>
#include <VMUtils/json_binding.hpp>

using namespace glm;
using namespace hydrant;

struct PagingRendererConfig : vm::json::Serializable<PagingRendererConfig>
{
	// VM_JSON_FIELD( IsosurfaceRenderMode, mode ) = IsosurfaceRenderMode::Color;
	VM_JSON_FIELD( std::size_t, mem_limit_mb ) = 1024 * 2;
};
