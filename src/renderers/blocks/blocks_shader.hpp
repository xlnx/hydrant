#pragma once

#include <VMUtils/enum.hpp>
#include <hydrant/core/glm_math.hpp>
#include <hydrant/core/shader.hpp>
#include <hydrant/bridge/sampler.hpp>
#include <hydrant/pixel_template.hpp>

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	VM_ENUM( BlocksRenderMode,
			 Volume, Solid );

	struct BlocksShader : IShader<StdVec4Pixel>
	{
		Sampler mean_tex;
		Sampler chebyshev;
		BlocksRenderMode render_mode = BlocksRenderMode::Volume;
		float density = 1e-2f;
	};
}

VM_END_MODULE()
