#pragma once

#include <hydrant/core/shader.hpp>

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	struct StdVec4Pixel : IPixel
	{
		void write_to( unsigned char dst[ 4 ] )
		{
			auto d = saturate( this->v );
			dst[ 0 ] = d.x;
			dst[ 1 ] = d.y;
			dst[ 2 ] = d.z;
			dst[ 3 ] = 255;
		}

	public:
		vec4 v;
	};
}

VM_END_MODULE()
