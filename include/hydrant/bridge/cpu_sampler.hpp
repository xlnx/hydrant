#pragma once

#include <VMUtils/concepts.hpp>
#include <cudafx/texture.hpp>
#include <hydrant/core/glm_math.hpp>

VM_BEGIN_MODULE( hydrant )

struct CpuSamplerUntyped : vm::NoCopy, vm::NoMove
{
public:
	template <typename T, typename E>
	T sample_3d_untyped( glm::vec<3, E> const &p ) const;
	template <typename T, typename E>
	T sample_2d_untyped( glm::vec<2, E> const &p ) const;
	template <typename T, typename E>
	T sample_1d_untyped( E const &x ) const;
};

VM_EXPORT
{
	template <typename T>
	struct CpuSampler : CpuSamplerUntyped
	{
		CpuSampler( T const *data, glm::uvec3 const &dim,
					cufx::Texture::Options const &opts ) :
		  data( data ), idim( dim ), fdim( dim ), opts( opts )
		{
		}

	public:
		T sample_3d( glm::vec3 const &p ) const
		{
			vec3 fp = opts.normalize_coords ? p * fdim : p;
			ivec3 ip = floor( fp );
			ip = clamp( ip, ivec3( 0 ), idim );
			return data[ ip.z * idim.x * idim.y +
						 ip.y * idim.x +
						 ip.x ];
		}

		T sample_2d( glm::vec2 const &p ) const
		{
			return 0;
		}

		T sample_1d( float const &x ) const
		{
			return 0;
		}

	private:
		T const *data;
		glm::ivec3 idim;
		glm::vec3 fdim;
		cufx::Texture::Options opts;
	};
}

template <typename T, typename E>
T CpuSamplerUntyped::sample_3d_untyped( glm::vec<3, E> const &p ) const
{
	return static_cast<CpuSampler<T> const *>( this )->sample_3d( glm::vec3( p ) );
}
template <typename T, typename E>
T CpuSamplerUntyped::sample_2d_untyped( glm::vec<2, E> const &p ) const
{
	return static_cast<CpuSampler<T> const *>( this )->sample_2d( glm::vec2( p ) );
}
template <typename T, typename E>
T CpuSamplerUntyped::sample_1d_untyped( E const &x ) const
{
	return static_cast<CpuSampler<T> const *>( this )->sample_1d( float( x ) );
}

VM_END_MODULE()
