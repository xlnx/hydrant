#pragma once

#include <cudafx/texture.hpp>
#include <hydrant/core/glm_math.hpp>
#include <hydrant/bridge/cpu_sampler.hpp>

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	struct Sampler
	{
		Sampler() = default;
		Sampler( cufx::Texture const &tex ) :
		  cu( tex.get() )
		{
		}
		Sampler( cudaTextureObject_t const &tex ) :
		  cu( tex )
		{
		}
		Sampler( ICpuSampler &sampler ) :
		  cpu( &sampler )
		{
		}

	public:
		template <typename T, typename E>
		__host__ __device__ T
		  sample_3d( glm::vec<3, E> const &p ) const
		{
#if CUFX_DEVICE_CODE
			return tex3D<T>( cu, p.x, p.y, p.z );
#else
			return cpu->sample_3d_untyped<T>( p );
#endif
		}
		template <typename T, typename E>
		__host__ __device__ T
		  sample_2d( glm::vec<2, E> const &p ) const
		{
#if CUFX_DEVICE_CODE
			return tex2D<T>( cu, p.x, p.y );
#else
			return cpu->sample_2d_untyped<T>( p );
#endif
		}
		template <typename T, typename E>
		__host__ __device__ T
		  sample_1d( E const &x ) const
		{
#if CUFX_DEVICE_CODE
			return tex1D<T>( cu, x );
#else
			return cpu->sample_1d_untyped<T>( x );
#endif
		}

	private:
		union
		{
			cudaTextureObject_t cu;
			ICpuSampler *cpu;
		};
	};
}

VM_END_MODULE()
