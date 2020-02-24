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
	// VM_ENUM( PagingRenderMode,
	// 		 Color, Normal, Position );

	struct PagingShader : IShader<StdVec4Pixel>
	{
		// PagingRenderMode mode;
		Sampler chebyshev;
		RtBlockPagingClient paging;
	};
}

VM_END_MODULE()
