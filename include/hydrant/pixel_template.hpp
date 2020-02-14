#pragma once

#include <hydrant/core/shader.hpp>

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	struct StdVec4Pixel : IPixel
	{
		void write_to( unsigned char dst[ 4 ] )
		{
			auto v = clamp( this->v * 255.f, vec4( 0.f ), vec4( 255.f ) );
			dst[ 0 ] = (unsigned char)( v.x );
			dst[ 1 ] = (unsigned char)( v.y );
			dst[ 2 ] = (unsigned char)( v.z );
			dst[ 3 ] = (unsigned char)( 255 );
		}

	public:
		vec4 v;
	};
}

VM_END_MODULE()
