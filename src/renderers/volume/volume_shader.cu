#include "volume.hpp"

VM_BEGIN_MODULE( hydrant )

struct VolumeShaderKernel : VolumeShader
{
	__host__ __device__ int
	  skip_nblock_steps( Ray &ray, vec3 const &ip,
						 int nblocks, float cdu, float step ) const
	{
		float tnear, tfar;
		ray.intersect( Box3D{ ip, ip + 1.f }, tnear, tfar );
		float di = ceil( ( tfar + ( nblocks - 1 ) * cdu ) / step );
		ray.o += ray.d * di * step;
		return (int)di;
	}

	__host__ __device__ void
	  main( Pixel &pixel_in_out ) const
	{
		const auto cdu = 1.f / compMax( abs( pixel_in_out.ray.d ) );
		const auto opacity_threshold = 0.95f;

		auto pixel = pixel_in_out;
		auto &ray = pixel.ray;
		auto &nsteps = pixel.nsteps;

		while ( nsteps > 0 ) {
			vec3 ip = floor( ray.o );
			if ( int cd = chebyshev.sample_3d<int>( ip ) ) {
				nsteps -= skip_nblock_steps( ray, ip, cd, cdu, step );
			} else {
				auto pgid = vaddr.sample_3d<int>( ip );
				if ( pgid != -1 ) {
					auto spl = block_sampler[ pgid ].sample_3d<float>( ray.o - ip );
					auto val = transfer_fn.sample_1d<vec4>( spl );
					auto col = val * density;
					pixel.v += col * ( 1.f - pixel.v.w );
					if ( pixel.v.w > opacity_threshold ) {
						break;
					}
				} else {
					// ivec3 ipm = mod( ip, vec3( MAXX ) );
					// absent_coord[ ipm.x ][ ipm.y ][ ipm.z ] = ip;
					// is_absent[ ipm.x ][ ipm.y ][ ipm.z ] = true;
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
  name( "volume_shader" )
	.cuda<VolumeShaderKernel>()
	.cpu<VolumeShaderKernel>(),
  VolumeShader );

VM_END_MODULE()
