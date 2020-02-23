#pragma once

#include <VMUtils/enum.hpp>
#include <hydrant/core/glm_math.hpp>
#include <hydrant/core/shader.hpp>
#include <hydrant/bridge/sampler.hpp>
#include <hydrant/pixel_template.hpp>
#include <hydrant/rt_block_paging.hpp>

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	VM_ENUM( IsosurfaceRenderMode,
			 Color, Normal, Position );

	struct IsosurfaceShader : IShader<StdVec4Pixel>
	{
		IsosurfaceRenderMode mode;
		mat4 to_world;
		vec3 eye_pos;
		vec3 light_pos;
		float isovalue;
		Sampler chebyshev;
		RtBlockPagingClient paging;
	};
}

VM_END_MODULE()
