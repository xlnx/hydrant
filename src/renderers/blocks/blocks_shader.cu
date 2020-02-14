#include "blocks_shader.hpp"

VM_BEGIN_MODULE( hydrant )

struct VisBlocksShaderKernel : BlocksShader
{
	__device__ int
	  skip_nblock_steps( Ray &ray, vec3 const &ip,
						 float nblocks, float cdu, float step ) const
	{
		float tnear, tfar;
		ray.intersect( Box3D{ ip, ip + 1.f }, tnear, tfar );
		float di = ceil( ( tfar + ( nblocks - 1 ) * cdu ) / step );
		ray.o += ray.d * di * step;
		return (int)di;
	}

	__device__ void
	  main( Pixel &pixel_in_out ) const
	{
		const auto cdu = 1.f / compMax( abs( pixel_in_out.ray.d ) );
		const auto opacity_threshold = 0.95f;

		auto pixel = pixel_in_out;
		auto &ray = pixel.ray;
		auto &nsteps = pixel.nsteps;

		while ( nsteps > 0 ) {
			vec3 ip = floor( ray.o );
			if ( float cd = chebyshev_tex.sample_3d<float>( ip ) ) {
				nsteps -= skip_nblock_steps( ray, ip, cd, cdu, step );
			} else {
				auto rip = clamp( ip, bbox.min, bbox.max - 1.f ) / ( bbox.max - bbox.min );
				if ( render_mode == BlocksRenderMode::BrmVolume ) {
					auto mean = mean_tex.sample_3d<float>( ip );
					auto col = vec4( rip, 1.f ) * density * mean;
					pixel.v += col * ( 1.f - pixel.v.w );
					if ( pixel.v.w > opacity_threshold ) {
						nsteps = 0;
						break;
					}
				} else if ( render_mode == BlocksRenderMode::BrmSolid ) {
					pixel.v = vec4( rip, 1.f );
					nsteps = 0;
					break;
				}
			}
			ray.o += ray.d * step;
			nsteps -= 1;
		}
		pixel_in_out = pixel;
	}
};

REGISTER_SHADER_BUILDER(
  name( "vis_blocks_shader" )
	.cuda<VisBlocksShaderKernel>(),
  BlocksShader );

VM_END_MODULE()
