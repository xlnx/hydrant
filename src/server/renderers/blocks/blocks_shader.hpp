#pragma once

#include <hydrant/core/glm_math.hpp>
#include <hydrant/core/shader.hpp>
#include <hydrant/bridge/sampler.hpp>
#include <hydrant/pixel_template.hpp>
#include <blocks.schema.hpp>

struct BlocksShader : IShader<StdVec4Pixel>
{
	Sampler mean_tex;
	Sampler chebyshev;
	BlocksRenderMode render_mode = BlocksRenderMode::Volume;
	float density = 1e-2f;
};
