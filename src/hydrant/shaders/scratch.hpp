#include <glm_math.hpp>
#include <texture_adapter.hpp>
#include "chebyshev_shader.hpp"

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	struct ScratchPixel
	{
		void write_to( unsigned char dst[ 4 ] )
		{
			auto v = glm::clamp( this->v * 255.f, glm::vec4( 0.f ), glm::vec4( 255.f ) );
			dst[ 0 ] = (unsigned char)( v.x );
			dst[ 1 ] = (unsigned char)( v.y );
			dst[ 2 ] = (unsigned char)( v.z );
			dst[ 3 ] = (unsigned char)( 255 );
		}

	public:
		glm::vec4 v;
	};

	struct ScratchIntegrator
	{
		using Pixel = ScratchPixel;

		__device__ bool
		  integrate( glm::vec3 const &p, glm::vec3 const &ip, ScratchPixel &pixel ) const
		{
			const auto opacity_threshold = 0.95f;
			const auto density = 3e-4f;

			auto val = glm::vec4( thumbnail_tex.sample_3d<float2>( ip ).x );
			auto col = glm::vec4( ip, 1 ) * val * density;
			pixel.v += col * ( 1.f - pixel.v.w );

			return pixel.v.y <= opacity_threshold;
		}

		__device__ float
		  chebyshev( glm::vec3 const &ip ) const
		{
			return thumbnail_tex.sample_3d<float2>( ip ).y;
		}

	public:
		TextureAdapter thumbnail_tex;
	};

	SHADER_DECL( ChebyshevShader<ScratchIntegrator> );
}

VM_END_MODULE()
