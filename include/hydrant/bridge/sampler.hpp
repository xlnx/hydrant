#pragma once

#include <cudafx/texture.hpp>
#include <hydrant/core/glm_math.hpp>
#include <hydrant/bridge/cpu_sampler.hpp>

VM_BEGIN_MODULE( hydrant )

template <typename T>
struct CudaVecType
{
	using type = T;

	static __host__ __device__ T to( T v )
	{
		return v;
	}
};

#define DEF_SAMPLE_VEC_N( T, N )                                \
	template <qualifier Q>                                      \
	struct CudaVecType<vec<N, T, Q>>                            \
	{                                                           \
		using type = T##N;                                      \
                                                                \
		static __host__ __device__                              \
		  vec<N, T, Q>                                          \
		  to( T##N v )                                          \
		{                                                       \
			return reinterpret_cast<vec<N, T, Q> const &>( v ); \
		}                                                       \
	}

#define DEF_SAMPLE_VEC( T )   \
	DEF_SAMPLE_VEC_N( T, 1 ); \
	DEF_SAMPLE_VEC_N( T, 2 ); \
	DEF_SAMPLE_VEC_N( T, 3 ); \
	DEF_SAMPLE_VEC_N( T, 4 )

DEF_SAMPLE_VEC( float );
DEF_SAMPLE_VEC( double );

DEF_SAMPLE_VEC( int );
DEF_SAMPLE_VEC( uint );

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
			using Helper = CudaVecType<T>;
			return Helper::to( tex3D<typename Helper::type>( cu, p.x, p.y, p.z ) );
#else
			return cpu->sample_3d_untyped<T>( p );
#endif
		}
		template <typename T, typename E>
		__host__ __device__ T
		  sample_2d( glm::vec<2, E> const &p ) const
		{
#if CUFX_DEVICE_CODE
			using Helper = CudaVecType<T>;
			return Helper::to( tex2D<typename Helper::type>( cu, p.x, p.y ) );
#else
			return cpu->sample_2d_untyped<T>( p );
#endif
		}
		template <typename T, typename E>
		__host__ __device__ T
		  sample_1d( E const &x ) const
		{
#if CUFX_DEVICE_CODE
			using Helper = CudaVecType<T>;
			return Helper::to( tex1D<typename Helper::type>( cu, x ) );
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
