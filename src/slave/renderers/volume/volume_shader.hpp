#pragma once

#include <VMUtils/enum.hpp>
#include <hydrant/core/glm_math.hpp>
#include <hydrant/core/shader.hpp>
#include <hydrant/bridge/sampler.hpp>
#include <hydrant/pixel_template.hpp>
#include <hydrant/paging/block_paging.hpp>
#include <volume.schema.hpp>

struct VolumePixel : StdVec4Pixel
{
	vec3 theta;
	float phi;
};

struct VolumeFetchPixel
{
	vec3 theta;
	float phi;
	vec4 val;
};

struct VolumeShader : IShader<VolumePixel>
{
	VolumeRenderMode mode;
	float rank;
	Sampler transfer_fn;
	Sampler chebyshev;
	BlockPaging paging;
};
