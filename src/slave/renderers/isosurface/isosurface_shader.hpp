#pragma once

#include <VMUtils/enum.hpp>
#include <hydrant/core/glm_math.hpp>
#include <hydrant/core/shader.hpp>
#include <hydrant/bridge/sampler.hpp>
#include <hydrant/pixel_template.hpp>
#include <hydrant/paging/block_paging.hpp>
#include <isosurface.schema.hpp>

struct IsosurfacePixel : StdVec4Pixel
{
	vec3 origin;
	float depth;
};

struct IsosurfaceFetchPixel
{
	uchar3 val;
	float depth;
};

struct IsosurfaceShader : IShader<IsosurfacePixel>
{
	IsosurfaceRenderMode mode;
	mat4 to_world;
	vec3 eye_pos;
	vec3 light_pos;
	vec3 surface_color;
	float isovalue;
	Sampler chebyshev;
	BlockPaging paging;
};
