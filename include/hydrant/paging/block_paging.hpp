#pragma once

#include <VMUtils/modules.hpp>
#include <VMUtils/attributes.hpp>
#include <hydrant/core/glm_math.hpp>

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

	struct BlockPaging
	{
	public:
		Sampler vaddr;
		int lowest_blkcnt;
		BlockSampler const *block_sampler;
	};
}

VM_END_MODULE()
