#include "paging_shader.hpp"

struct PagingShaderKernel : PagingShader
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
	  fetch( Pixel const &pixel_in, void *pixel_out_ ) const
	{
		auto pixel_out = reinterpret_cast<PagingFetchPixel *>( pixel_out_ );
		auto val = saturate( pixel_in.v );
		pixel_out->val = uchar3{ val.x, val.y, val.z };
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
				auto pgid = paging.vaddr.sample_3d<int>( ip );
				if ( pgid != -1 ) {
					vec4 col;
					if ( pgid >= paging.lowest_blkcnt ) {
						col = vec4( 0, 1, 0, 1 ) * .5f;
					} else {
						col = vec4( 1, 0, 0, 1 ) * .2f;
					}
					nsteps -= skip_nblock_steps( ray, ip, 1, cdu, step );
					pixel.v += col * ( 1.f - pixel.v.w );
					if ( pixel.v.w > opacity_threshold ) {
						break;
					}
				} else {
					/* page fault */
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
  name( "paging_shader" )
	.cuda<PagingShaderKernel>()
	.cpu<PagingShaderKernel>(),
  PagingShader );
