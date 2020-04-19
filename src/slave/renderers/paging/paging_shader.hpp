#pragma once

#include <VMUtils/enum.hpp>
#include <hydrant/core/glm_math.hpp>
#include <hydrant/core/shader.hpp>
#include <hydrant/bridge/sampler.hpp>
#include <hydrant/pixel_template.hpp>
#include <hydrant/paging/block_paging.hpp>
#include "paging.schema.hpp"

struct PagingShader : IShader<StdVec4Pixel>
{
	// PagingRenderMode mode;
	Sampler chebyshev;
	BlockPaging paging;
};
