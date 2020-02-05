#pragma once

#include <vector>
#include <VMUtils/modules.hpp>
#include <glm_math.hpp>

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	template <typename T>
	struct Buffer3D
	{
		Buffer3D( uvec3 const &dim ) :
		  d( dim ),
		  dx( 1, dim.x, dim.x * dim.y ),
		  buf( dim.x * dim.y * dim.z )
		{
		}

	public:
		T const &operator[]( uvec3 const &idx ) const { return buf[ dot( idx, dx ) ]; }
		T &operator[]( uvec3 const &idx ) { return buf[ dot( idx, dx ) ]; }
		T const *data() const { return buf.data(); }
		T *data() { return buf.data(); }
		uvec3 const &dim() const { return d; }

	private:
		uvec3 d, dx;
		std::vector<T> buf;
	};
}

VM_END_MODULE()
