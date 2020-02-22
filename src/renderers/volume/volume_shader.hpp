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
	VM_ENUM( VolumeRenderMode,
			 Volume, DebCache );

	struct VolumeShader : IShader<StdVec4Pixel>
	{
		VolumeRenderMode mode;
		float density;
		Sampler transfer_fn;
		Sampler chebyshev;
		RtBlockPagingClient paging;
	};
}

VM_END_MODULE()
