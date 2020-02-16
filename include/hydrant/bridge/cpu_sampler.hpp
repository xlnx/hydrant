#pragma once

#include <VMUtils/concepts.hpp>
#include <cudafx/texture.hpp>
#include <hydrant/core/glm_math.hpp>

VM_BEGIN_MODULE( hydrant )

struct ICpuSampler
{
public:
	template <typename T, typename E>
	T sample_3d_untyped( glm::vec<3, E> const &p ) const;
	template <typename T, typename E>
	T sample_2d_untyped( glm::vec<2, E> const &p ) const;
	template <typename T, typename E>
	T sample_1d_untyped( E x ) const;
};

VM_EXPORT
{
	template <typename T>
	struct CpuSampler : ICpuSampler
	{
		CpuSampler( glm::uvec3 const &dim,
					cufx::Texture::Options const &opts ) :
		  idim( dim ), fdim( dim ), opts( opts )
		{
		}

	public:
		CpuSampler &source( T const *data )
		{
			this->data = data;
			return *this;
		}

		T sample_3d( glm::vec3 const &p ) const
		{
			vec3 fp = opts.normalize_coords ? p * fdim : p;
			ivec3 ip = floor( fp );
			ip = clamp( ip, ivec3( 0 ), idim - 1 );
			return data[ ip.z * idim.x * idim.y +
						 ip.y * idim.x +
						 ip.x ];
		}

		T sample_2d( glm::vec2 const &p ) const
		{
			vec2 fp = opts.normalize_coords ? p * vec2( fdim.x, fdim.y ) : p;
			ivec2 ip = floor( fp );
			ip = clamp( ip, ivec2( 0 ), ivec2( idim.x - 1, idim.y - 1 ) );
			return data[ ip.y * idim.x +
						 ip.x ];
		}

		T sample_1d( float x ) const
		{
			float fx = opts.normalize_coords ? x * fdim.x : x;
			if ( opts.filter_mode == cufx::Texture::FilterMode::Linear ) {
				int ix = floor( fx ), jx = ceil( fx );
				float k = fx - ix;
				ix = clamp( ix, 0, idim.x - 1 );
				jx = clamp( jx, 0, idim.x - 1 );
				return data[ ix ] * ( 1 - k ) + data[ jx ] * k;
			} else {
				int ix = floor( fx );
				ix = clamp( ix, 0, idim.x - 1 );
				return data[ ix ];
			}
		}

	private:
		T const *data = nullptr;
		glm::ivec3 idim;
		glm::vec3 fdim;
		cufx::Texture::Options opts;
	};
}

template <typename T, typename E>
T ICpuSampler::sample_3d_untyped( glm::vec<3, E> const &p ) const
{
	return static_cast<CpuSampler<T> const *>( this )->sample_3d( glm::vec3( p ) );
}
template <typename T, typename E>
T ICpuSampler::sample_2d_untyped( glm::vec<2, E> const &p ) const
{
	return static_cast<CpuSampler<T> const *>( this )->sample_2d( glm::vec2( p ) );
}
template <typename T, typename E>
T ICpuSampler::sample_1d_untyped( E x ) const
{
	return static_cast<CpuSampler<T> const *>( this )->sample_1d( float( x ) );
}

VM_END_MODULE()
