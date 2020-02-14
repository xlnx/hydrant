#pragma once

#include <hydrant/glm_math.hpp>
#include <hydrant/shader.hpp>
#include <hydrant/texture_adapter.hpp>

#define MAX_CACHE_SIZE ( 64 )

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	struct StandardVolumePixel
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
		// struct Record
		// {
		// 	vec4 v, u;
		// 	vec3 p;
		// 	int n;
		// };

		// Record rem[ MAX_REMAINING ];
		vec4 v;
		Ray ray;
		int nsteps;
	};

	struct VolumeShader
	{
		using Pixel = StandardVolumePixel;

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
		  raymarch( Pixel &pixel_in_out ) const
		{
#define MAXX ( 8 )
			__shared__ uvec3 absent_coord[ MAXX ][ MAXX ][ MAXX ];
			__shared__ bool is_absent[ MAXX ][ MAXX ][ MAXX ];
			__shared__ int wg_emit_cnt;

			bool is_worker = threadIdx.x < MAXX && threadIdx.y < MAXX;
			if ( is_worker ) {
				for ( int i = 0; i != MAXX; ++i ) {
					is_absent[ threadIdx.y ][ threadIdx.x ][ i ] = false;
				}
			}
			wg_emit_cnt = 0;

			__syncthreads();

			const auto cdu = 1.f / compMax( abs( pixel_in_out.ray.d ) );
			const auto opacity_threshold = 0.95f;
			const auto density = 1e-1f;

			auto pixel = pixel_in_out;
			auto &ray = pixel.ray;
			auto &nsteps = pixel.nsteps;

			while ( nsteps > 0 ) {
				vec3 ip = floor( ray.o );
				if ( float cd = chebyshev_tex.sample_3d<float>( ip ) ) {
					nsteps -= skip_nblock_steps( ray, ip, cd, cdu, step );
				} else {
					auto present_id = present_tex.sample_3d<int>( ip );
					if ( present_id != -1 ) {
						auto pt = ( cache_du.x + ( ray.o - ip ) ) * cache_du.y;
						auto val = vec4( cache_tex[ present_id ].sample_3d<float>( pt ) );
						// auto val = vec4( ray.o - ip, 1 );
						// auto val = vec4( chebyshev_tex.sample_3d<float2>( ip ).x );
						auto col = val * density;
						pixel.v += col * ( 1.f - pixel.v.w );
						if ( pixel.v.w > opacity_threshold ) {
							break;
						}
					} else {
						ivec3 ipm = mod( ip, vec3( MAXX ) );
						absent_coord[ ipm.x ][ ipm.y ][ ipm.z ] = ip;
						is_absent[ ipm.x ][ ipm.y ][ ipm.z ] = true;
						break;
					}
				}
				ray.o += ray.d * step;
				nsteps -= 1;
			}
			pixel_in_out = pixel;

			__syncthreads();

			int wg_id = blockIdx.x + blockIdx.y * blockDim.x;
			char *wg_base_ptr = absent_buf.ptr() + wg_len_bytes * wg_id;
			int *wg_emit_cnt_ptr = (int *)wg_base_ptr;
			uvec3 *wg_ptr = (uvec3 *)( wg_base_ptr + sizeof( int ) );

			if ( is_worker ) {
				for ( int i = 0; i != MAXX; ++i ) {
					if ( is_absent[ threadIdx.y ][ threadIdx.x ][ i ] ) {
						auto old = atomicAdd( &wg_emit_cnt, 1 );
						if ( old < wg_max_emit_cnt ) {
							wg_ptr[ old ] = absent_coord[ threadIdx.y ][ threadIdx.x ][ i ];
						}
					}
				}
			}

			__syncthreads();

			if ( threadIdx.x == 0 && threadIdx.y == 0 ) {
				*wg_emit_cnt_ptr = min( wg_emit_cnt, wg_max_emit_cnt );
			}
		}

	public:
		Box3D bbox;
		float step;
		int max_steps = 500;

		cufx::MemoryView1D<char> absent_buf;
		int wg_max_emit_cnt;
		int wg_len_bytes;

		vec2 cache_du;
		TextureAdapter chebyshev_tex;
		TextureAdapter present_tex;
		TextureAdapter cache_tex[ MAX_CACHE_SIZE ];
	};

	struct VolumePixelShader : VolumeShader, PixelShader
	{
		__device__ void
		  apply( Pixel &pixel_in_out ) const
		{
			auto pixel = pixel_in_out;
			// if ( pixel.nsteps ) {
			this->raymarch( pixel );
			pixel_in_out = pixel;
			// }
		}
	};

	struct VolumeRayEmitShader : VolumeShader, RayEmitShader
	{
		__device__ void
		  apply( Ray const &ray_in, Pixel &pixel_out ) const
		{
			auto ray = ray_in;
			Pixel pixel = {};
			float tnear, tfar;
			if ( ray.intersect( bbox, tnear, tfar ) ) {
				ray.o += ray.d * tnear;
				pixel.nsteps = min( max_steps, int( ( tfar - tnear ) / this->step ) );
				pixel.ray = ray;
				this->raymarch( pixel );
			} else {
				pixel.nsteps = 0;
			}
			pixel_out = pixel;
		}
	};
}

VM_END_MODULE()
