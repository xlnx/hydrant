#pragma once

#include <cudafx/texture.hpp>
#include <hydrant/core/glm_math.hpp>

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	struct Sampler
	{
		Sampler &operator=( cufx::Texture const &tex )
		{
			cu = tex.get();
			return *this;
		}
		Sampler &operator=( cudaTextureObject_t const &tex )
		{
			cu = tex;
			return *this;
		}

	public:
		template <typename T, typename E>
		__host__ __device__ T
		  sample_3d( glm::vec<3, E> const &p ) const
		{
#ifdef __CUDACC__
			return tex3D<T>( cu, p.x, p.y, p.z );
#else
#endif
		}
		template <typename T, typename E>
		__host__ __device__ T
		  sample_2d( glm::vec<2, E> const &p ) const
		{
#ifdef __CUDACC__
			return tex2D<T>( cu, p.x, p.y );
#else
#endif
		}
		template <typename T, typename E>
		__host__ __device__ T
		  sample_1d( E const &x ) const
		{
#ifdef __CUDACC__
			return tex1D<T>( cu, x );
#else
#endif
		}

	private:
		union
		{
			cudaTextureObject_t cu;
			void *in;
		};
	};
}

VM_END_MODULE()
