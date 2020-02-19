#pragma once

#include <hydrant/core/shader.hpp>

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	struct StdVec4Pixel : IPixel
	{
		__host__ __device__ void
		  write_to( uchar4 &dst ) const
		{
			reinterpret_cast<tvec4<unsigned char> &>( dst ) = saturate( this->v );
			dst.w = 255;
		}

	public:
		vec4 v;
	};
}

VM_END_MODULE()
