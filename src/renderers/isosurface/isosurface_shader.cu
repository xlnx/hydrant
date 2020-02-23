#include "isosurface_shader.hpp"

VM_BEGIN_MODULE( hydrant )

struct IsosurfaceShaderKernel : IsosurfaceShader
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

	/* central difference */
	__host__ __device__ vec3
	  gradient( BlockSampler const &s, vec3 p, float dt ) const
	{
		return vec3( s.sample_3d<float>( p - vec3( dt, 0, 0 ) ) - s.sample_3d<float>( p + vec3( dt, 0, 0 ) ),
					 s.sample_3d<float>( p - vec3( 0, dt, 0 ) ) - s.sample_3d<float>( p + vec3( 0, dt, 0 ) ),
					 s.sample_3d<float>( p - vec3( 0, 0, dt ) ) - s.sample_3d<float>( p + vec3( 0, 0, dt ) ) );
	}

	__host__ __device__ void
	  main( Pixel &pixel_in_out ) const
	{
		auto pixel = pixel_in_out;
		auto &ray = pixel.ray;
		auto &nsteps = pixel.nsteps;

		const auto cdu = 1.f / compMax( abs( pixel_in_out.ray.d ) );
		float prev_value = 0.0;
		// pixel.v = vec4( 1 );

		while ( nsteps > 0 ) {
			vec3 ip = floor( ray.o );
			if ( int cd = chebyshev.sample_3d<int>( ip ) ) {
				nsteps -= skip_nblock_steps( ray, ip, cd, cdu, step );
			} else {
				auto pgid = paging.vaddr.sample_3d<int>( ip );
				if ( pgid != -1 ) {
					auto &sampler = paging.block_sampler[ pgid ];
					auto value = sampler.sample_3d<float>( ray.o - ip );
					if ( sign( value - isovalue ) != sign( prev_value - isovalue ) ) {
						auto &p = ray.o;
						auto &d = ray.d;
						auto &dt = step;

						pixel.v = vec4( 0.78, 0.13, 0.13, 1.0 );

						/* linear approximation of intersection point */
						vec3 prev_p = p - dt * d;
						float a = ( isovalue - prev_value ) / ( value - prev_value );
						vec3 inter_p = ( 1.f - a ) * ( p - dt * d ) + a * p;
						/* TODO: sample at different dt for each axis to avoid having undo scaling */
						vec3 nn = gradient( sampler, inter_p - ip, dt );

						/* TODO: can we optimize somehow? */
						auto &world_p = inter_p;
						vec3 n = normalize( nn );
						vec3 light_dir = normalize( vec3( 0, 10, 0 ) - world_p );
						vec3 h = normalize( light_dir - world_p ); /* eye is at origin */
						const float ambient = 0.2;
						float diffuse = .6f * clamp( dot( light_dir, n ), 0.f, 1.f );
						float specular = .2f * pow( clamp( dot( h, n ), 0.f, 1.f ), 100.f );
						// float distance = length( world_p ); /* eye is at origin */

						pixel.v = pixel.v * ( ambient + ( diffuse + specular ) );

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
  name( "isosurface_shader" )
	.cuda<IsosurfaceShaderKernel>()
	.cpu<IsosurfaceShaderKernel>(),
  IsosurfaceShader );

VM_END_MODULE()
