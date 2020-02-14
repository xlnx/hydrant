#pragma once

#include <hydrant/core/glm_math.hpp>
#include <hydrant/core/shader.hpp>
#include <hydrant/core/sampler.hpp>
#include <hydrant/pixel_template.hpp>

#define MAX_CACHE_SIZE ( 64 )

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	struct VolumeShader : IShader<StdVec4Pixel>
	{
		cufx::MemoryView1D<char> absent_buf;
		int wg_max_emit_cnt;
		int wg_len_bytes;

		vec2 cache_du;
		Sampler chebyshev_tex;
		Sampler present_tex;
		Sampler cache_tex[ MAX_CACHE_SIZE ];
	};
}

VM_END_MODULE()
