#pragma once

#include <VMUtils/enum.hpp>
#include <hydrant/core/glm_math.hpp>
#include <hydrant/core/shader.hpp>
#include <hydrant/bridge/sampler.hpp>
#include <hydrant/pixel_template.hpp>
#include <hydrant/paging/block_paging.hpp>
#include <volume.schema.hpp>

struct VolumeShader : IShader<StdVec4Pixel>
{
	float density;
	Sampler transfer_fn;
	Sampler chebyshev;
	BlockPaging paging;
};
