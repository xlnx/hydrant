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

		T sample_3d( glm::vec3 const &x ) const { return sample_impl( x ); }

		T sample_2d( glm::vec2 const &x ) const { return sample_impl( x ); }

		T sample_1d( float x ) const { return sample_impl( x ); }

	private:
		template <typename U>
		void get_dim( U &dst ) const
		{
			dst = fdim;
		}

		void get_dim( float &dst ) const
		{
			dst = fdim.x;
		}

		T visit( glm::ivec3 const &ix ) const
		{
			return data[ ix.z * idim.x * idim.y +
						 ix.y * idim.x +
						 ix.x ];
		}
		T visit( glm::ivec2 const &ix ) const
		{
			return data[ ix.y * idim.x +
						 ix.x ];
		}
		T visit( int ix ) const
		{
			return data[ ix ];
		}

		T filter_none( glm::vec3 const &fx ) const
		{
			auto ix = clamp( ivec3( floor( fx ) ), ivec3( 0 ), idim - 1 );
			return visit( ix );
		}
		T filter_none( glm::vec2 const &fx ) const
		{
			auto ix = clamp( ivec2( floor( fx ) ), ivec2( 0 ), ivec2( idim ) - 1 );
			return visit( ix );
		}
		T filter_none( float fx ) const
		{
			auto ix = clamp( int( floor( fx ) ), 0, idim.x - 1 );
			return visit( ix );
		}

		T filter_linear( glm::vec3 const &fx ) const
		{
			auto flr = floor( fx );
			auto a = fx - flr;
			auto ix = clamp( ivec3( flr ), ivec3( 0 ), idim - 1 );
			auto jx = clamp( ivec3( ceil( fx ) ), ivec3( 0 ), idim - 1 );

			auto x0_0 = do_lerp( visit( { ix.x, ix.y, ix.z } ), visit( { jx.x, ix.y, ix.z } ), a.x );
			auto x1_0 = do_lerp( visit( { ix.x, jx.y, ix.z } ), visit( { jx.x, jx.y, ix.z } ), a.x );
			auto x0_1 = do_lerp( visit( { ix.x, ix.y, jx.z } ), visit( { jx.x, ix.y, jx.z } ), a.x );
			auto x1_1 = do_lerp( visit( { ix.x, jx.y, jx.z } ), visit( { jx.x, jx.y, jx.z } ), a.x );

			auto y0 = do_lerp( x0_0, x1_0, a.y );
			auto y1 = do_lerp( x0_1, x1_1, a.y );

			return do_lerp( y0, y1, a.z );
		}
		T filter_linear( glm::vec2 const &fx ) const
		{
			auto flr = floor( fx );
			auto a = fx - flr;
			auto ix = clamp( ivec2( flr ), ivec2( 0 ), ivec2( idim ) - 1 );
			auto jx = clamp( ivec2( ceil( fx ) ), ivec2( 0 ), ivec2( idim ) - 1 );

			auto x0 = do_lerp( visit( { ix.x, ix.y } ), visit( { jx.x, ix.y } ), a.x );
			auto x1 = do_lerp( visit( { ix.x, jx.y } ), visit( { jx.x, jx.y } ), a.x );

			return do_lerp( x0, x1, a.y );
		}
		T filter_linear( float fx ) const
		{
			auto flr = floor( fx );
			auto a = fx - flr;
			auto ix = clamp( int( flr ), 0, idim.x - 1 );
			auto jx = clamp( int( ceil( fx ) ), 0, idim.x - 1 );
			return do_lerp( visit( ix ), visit( jx ), a );
		}

		T do_lerp( T const &x, T const &y, float a ) const
		{
			return x * ( 1.f - a ) + y * a;
		}

		template <typename U>
		bool in_bounds( U const &fx, U const &fd ) const
		{
			return !( glm::any( lessThan( fx, U( 0 ) ) ) ||
					  glm::any( greaterThan( fx, fd ) ) );
		}
		bool in_bounds( float fx, float fd ) const
		{
			return 0.f <= fx && fx <= fd;
		}

		template <typename U>
		T sample_impl( U const &x ) const
		{
			U fd;
			get_dim( fd );
			auto fx = opts.normalize_coords ? fd * x : x;
			switch ( opts.address_mode ) {
			case cufx::Texture::AddressMode::Clamp:
				fx = clamp( fx, U( 0 ), fd );
				break;
			case cufx::Texture::AddressMode::Border:
				if ( !in_bounds( fx, fd ) ) return T( 0 );
				break;
			case cufx::Texture::AddressMode::Mirror:
			case cufx::Texture::AddressMode::Wrap:
				fx = mod( fx, fd );
				break;
			}
			switch ( opts.filter_mode ) {
			case cufx::Texture::FilterMode::None:
				return filter_none( fx );
			case cufx::Texture::FilterMode::Linear:
				return filter_linear( fx - .5f );
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
