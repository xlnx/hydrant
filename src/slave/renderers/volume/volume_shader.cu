#include "volume_shader.hpp"

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
	  init( Pixel &pixel_out, Ray const &ray ) const
	{
		pixel_out.theta = vec3( 0 );
		pixel_out.phi = 1.f;
	}

	__host__ __device__ void
	  fetch( Pixel const &pixel_in, void *pixel_out_ ) const
	{
		auto pixel_out = reinterpret_cast<VolumeFetchPixel *>( pixel_out_ );
		pixel_out->val = pixel_in.v;
		pixel_out->theta = pixel_in.theta;
		pixel_out->phi = pixel_in.phi;
	}

	__host__ __device__ void
	  main( Pixel &pixel_in_out ) const
	{
		const auto cdu = 1.f / compMax( abs( pixel_in_out.ray.d ) );
		// const auto opacity_threshold = 0.999f;

		auto pixel = pixel_in_out;
		auto &ray = pixel.ray;
		auto &nsteps = pixel.nsteps;
		auto &step_size = pixel.step_size;

		while ( nsteps > 0 ) {
			vec3 ip = floor( ray.o );
			if ( int cd = chebyshev.sample_3d<int>( ip ) ) {
			    // skip_block.b_i = 0
				nsteps -= skip_nblock_steps( ray, ip, cd, cdu, step * step_size );
			} else {
				auto pgid = paging.vaddr.sample_3d<int>( ip );
				if ( pgid == -1 ) break;
				
				auto s_i = paging.block_sampler[ pgid ].sample_3d<float>( ray.o - ip );
				auto ub_i = transfer_fn.sample_1d<vec4>( s_i );
				if ( mode == VolumeRenderMode::Partition ) {
				    vec3 lower = { 1, 0, 0 };
				    vec3 upper = { 0, 0, 1 };
					auto v = mix( lower, upper, rank ) *
						       float( length( vec3( ub_i ) ) );
					ub_i = vec4( v.x, v.y, v.z, ub_i.w );
				} else if ( mode == VolumeRenderMode::Paging ) {
					if ( pgid >= paging.lowest_blkcnt ) {
						ub_i = vec4( 0, 1, 0, ub_i.w );
					} else {
						ub_i = vec4( 1, 0, 0, ub_i.w );
					}
				}
				ub_i *= vec4( ub_i.w, ub_i.w, ub_i.w, 1 );
				pixel.theta += vec3( ub_i ) * pixel.phi;
				pixel.phi *= 1.f - ub_i.w;
				pixel.v += ub_i * ( 1.f - pixel.v.w );
				if ( pixel.v.w > 0.93 ) {
				    step_size = 16;
				} else if ( pixel.v.w > 0.85 ) {
				    step_size = 4;
				}
			}
			ray.o += ray.d * step * float( step_size );
			nsteps -= step_size;
		}
		pixel_in_out = pixel;
	}
};

REGISTER_SHADER_BUILDER(
  name( "volume_shader" )
	.cuda<VolumeShaderKernel>()
	.cpu<VolumeShaderKernel>(),
  VolumeShader );
