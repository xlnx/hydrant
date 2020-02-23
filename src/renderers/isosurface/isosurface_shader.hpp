#pragma once

#include <hydrant/core/glm_math.hpp>
#include <hydrant/core/shader.hpp>
#include <hydrant/bridge/sampler.hpp>
#include <hydrant/pixel_template.hpp>
#include <hydrant/rt_block_paging.hpp>

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	struct IsosurfaceShader : IShader<StdVec4Pixel>
	{
		float isovalue;
		Sampler chebyshev;
		RtBlockPagingClient paging;
	};
}

VM_END_MODULE()
