#pragma once

#include <hydrant/core/glm_math.hpp>
#include <hydrant/core/shader.hpp>
#include <hydrant/core/sampler.hpp>
#include <hydrant/pixel_template.hpp>

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	enum BlocksRenderMode
	{
		BrmVolume,
		BrmSolid
	};

	struct BlocksShader : IShader<StdVec4Pixel>
	{
		Sampler mean_tex;
		Sampler chebyshev_tex;
		BlocksRenderMode render_mode = BlocksRenderMode::BrmVolume;
		float density = 1e-2f;
	};
}

VM_END_MODULE()
