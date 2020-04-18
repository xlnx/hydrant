#pragma once

#include <hydrant/core/shader.hpp>

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	struct StdVec4Pixel : IPixel
	{
		__host__ __device__ void
		  write_to( uchar3 &dst, uchar3 const &clear_color ) const
		{
			if ( this->v.a ) {
				auto val = saturate( this->v );
				dst = uchar3{ val.x, val.y, val.z };
			} else {
				dst = clear_color;
			}
		}

	public:
		vec4 v;
	};
}

VM_END_MODULE()
