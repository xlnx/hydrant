#pragma once

#include <VMUtils/enum.hpp>
#include <VMUtils/json_binding.hpp>
#include <hydrant/core/glm_math.hpp>

using namespace glm;
using namespace hydrant;

VM_ENUM( IsosurfaceRenderMode,
		 Color, Normal, Position );

struct IsosurfaceRendererParams : vm::json::Serializable<IsosurfaceRendererParams>
{
	VM_JSON_FIELD( IsosurfaceRenderMode, mode ) = IsosurfaceRenderMode::Color;
	VM_JSON_FIELD( vec3, surface_color ) = { 1.f, 1.f, 1.f };
	VM_JSON_FIELD( float, isovalue ) = 0.5f;
	VM_JSON_FIELD( std::size_t, mem_limit_mb ) = 1024 * 2;
};
