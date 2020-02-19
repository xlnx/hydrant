#pragma once

#include <hydrant/core/glm_math.hpp>
#include <hydrant/core/shader.hpp>
#include <hydrant/bridge/sampler.hpp>
#include <hydrant/pixel_template.hpp>

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	struct BlockSamplerMapping
	{
		__host__ __device__ vec3
		  mapped( vec3 const &x ) const
		{
			return k * x + b;
		}

	public:
		VM_DEFINE_ATTRIBUTE( float, k );
		VM_DEFINE_ATTRIBUTE( vec3, b );
	};

	struct BlockSampler
	{
		template <typename T>
		__host__ __device__ T
		  sample_3d( vec3 const &x ) const
		{
			return sampler.sample_3d<T>( mapping.mapped( x ) );
		}

	public:
		VM_DEFINE_ATTRIBUTE( Sampler, sampler );
		VM_DEFINE_ATTRIBUTE( BlockSamplerMapping, mapping );
	};

	struct VolumeShader : IShader<StdVec4Pixel>
	{
		float density;
		Sampler transfer_fn;
		Sampler chebyshev;
		Sampler present;
		BlockSampler const *block_sampler;
	};
}

VM_END_MODULE()
