#include <glm_math.hpp>
#include "chebyshev_shader.hpp"

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	struct ScratchPixel
	{
		void write_to( unsigned char dst[ 4 ] )
		{
			auto v = glm::clamp( this->v.x * 255.f, 0.f, 255.f );
			dst[ 0 ] = (unsigned char)( v );
			dst[ 1 ] = (unsigned char)( v );
			dst[ 2 ] = (unsigned char)( v );
			dst[ 3 ] = (unsigned char)( 255 );
		}

	public:
		glm::vec2 v;
	};

	struct ScratchIntegrator
	{
		using Pixel = ScratchPixel;

		__host__ __device__ bool
		  integrate( glm::vec3 const &p, glm::ivec3 const &ip, ScratchPixel &pixel ) const
		{
			const auto opacity_threshold = 0.95f;
			const auto density = 3e-3f;

			auto val = thumbnail[ ip.x ][ ip.y ][ ip.z ].x;
			auto col = glm::vec2{ 1, 1 } * val * density;
			pixel.v += col * ( 1.f - pixel.v.y );

			return pixel.v.y <= opacity_threshold;
		}

		__host__ __device__ float
		  chebyshev( glm::ivec3 const &ip ) const
		{
			return thumbnail[ ip.x ][ ip.y ][ ip.z ].y;
		}

	public:
		glm::vec2 thumbnail[ 5 ][ 5 ][ 5 ];
	};

	SHADER_DECL( ChebyshevShader<ScratchIntegrator> );
}

VM_END_MODULE()
