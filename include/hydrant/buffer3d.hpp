#pragma once

#include <vector>
#include <VMUtils/modules.hpp>
#include <hydrant/core/glm_math.hpp>

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	template <typename T>
	struct Buffer3D
	{
		Buffer3D( uvec3 const &dim, T const &e = T() ) :
		  d( dim ),
		  dx( 1, dim.x, dim.x * dim.y ),
		  buf( dim.x * dim.y * dim.z, e )
		{
		}

	public:
		T const &operator[]( uvec3 const &idx ) const
		{
			return buf[ idx.x + idx.y * dx.y + idx.z * dx.z ];
		}
		T &operator[]( uvec3 const &idx )
		{
			return buf[ idx.x + idx.y * dx.y + idx.z * dx.z ];
		}

		T const *data() const { return buf.data(); }
		T *data() { return buf.data(); }

		uvec3 const &dim() const { return d; }
		std::size_t bytes() const { return d.x * d.y * d.z * sizeof( T ); }

		template <typename F>
		void iterate_3d( F const &f )
		{
			uvec3 idx;
			for ( idx.z = 0; idx.z != d.z; ++idx.z ) {
				for ( idx.y = 0; idx.y != d.y; ++idx.y ) {
					for ( idx.x = 0; idx.x != d.x; ++idx.x ) {
						f( idx );
					}
				}
			}
		}

	private:
		uvec3 d, dx;
		std::vector<T> buf;
	};
}

VM_END_MODULE()
