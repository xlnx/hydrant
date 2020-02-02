#pragma once

#include <vector>
#include <VMUtils/modules.hpp>
#include <VMUtils/attributes.hpp>
#include <varch/utils/io.hpp>
#include <varch/utils/common.hpp>
#include <glm_math.hpp>

VM_BEGIN_MODULE( hydrant )

using namespace glm;

VM_EXPORT
{
	struct ThumbUnit
	{
		float value;
		float chebyshev = 0;
	};

	template <typename T, typename = typename std::enable_if<
							std::is_base_of<ThumbUnit, T>::value>::type>
	struct Thumbnail
	{
		Thumbnail( vol::Idx const &dim ) :
		  dim( dim ),
		  buf( dim.total() )
		{
		}
		Thumbnail( vol::Reader &reader )
		{
			reader.read_typed( dim );
			reader.read_typed( buf );
		}

	public:
		T *data() { return buf.data(); }
		T const *data() const { return buf.data(); }

		T &operator[]( vol::Idx const &idx )
		{
			return buf[ idx.z * dim.x * dim.y +
						idx.y * dim.x +
						idx.x ];
		}

		T const &operator[]( vol::Idx const &idx ) const
		{
			return buf[ idx.z * dim.x * dim.y +
						idx.y * dim.x +
						idx.x ];
		}

		template <typename F>
		void iterate_3d( F const &f )
		{
			vol::Idx idx;
			for ( idx.z = 0; idx.z != dim.z; ++idx.z ) {
				for ( idx.y = 0; idx.y != dim.y; ++idx.y ) {
					for ( idx.x = 0; idx.x != dim.x; ++idx.x ) {
						f( idx );
					}
				}
			}
		}

		void compute_chebyshev()
		{
			auto maxd = max( dim.x, max( dim.y, dim.z ) );
			vec<3, int> d14[] = {
				{ -1, 0, 0 },
				{ 1, 0, 0 },
				{ 0, -1, 0 },
				{ 0, 1, 0 },
				{ 0, 0, -1 },
				{ 0, 0, 1 },
				{ -1, -1, -1 },
				{ -1, 1, -1 },
				{ 1, 1, -1 },
				{ 1, -1, -1 },
				{ -1, -1, 1 },
				{ -1, 1, 1 },
				{ 1, 1, 1 },
				{ 1, -1, 1 },
			};
			iterate_3d(
			  [&]( vol::Idx const &idx ) {
				  if ( !( *this )[ idx ].value ) {
					  ( *this )[ idx ].chebyshev = maxd;
				  }
			  } );
			bool update;
			do {
				update = false;
				iterate_3d(
				  [&]( vol::Idx const &idx ) {
					  if ( !( *this )[ idx ].value ) {
						  vec<3, int> u = { idx.x, idx.y, idx.z };
						  for ( int i = 0; i != 14; ++i ) {
							  auto v = u + d14[ i ];
							  float d0;
							  if ( v.x < 0 || v.y < 0 || v.z < 0 ||
								   v.x >= dim.x || v.y >= dim.y || v.z >= dim.z ) {
								  d0 = maxd;
							  } else {
								  d0 = ( *this )[ { v.x, v.y, v.z } ].chebyshev + 1.f;
							  }
							  if ( d0 < ( *this )[ idx ].chebyshev ) {
								  ( *this )[ idx ].chebyshev = d0;
								  update = true;
							  }
						  }
					  }
				  } );
			} while ( update );
		}

		void dump( vol::Writer &writer ) const
		{
			writer.write_typed( dim );
			writer.write_typed( buf );
		}

	public:
		VM_DEFINE_ATTRIBUTE( vol::Idx, dim );

	private:
		std::vector<T> buf;
	};
}

VM_END_MODULE()
