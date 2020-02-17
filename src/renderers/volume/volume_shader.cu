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
		// #define MAXX ( 8 )
		// 		__shared__ uvec3 absent_coord[ MAXX ][ MAXX ][ MAXX ];
		// 		__shared__ bool is_absent[ MAXX ][ MAXX ][ MAXX ];
		// 		__shared__ int wg_emit_cnt;

		// 		bool is_worker = threadIdx.x < MAXX && threadIdx.y < MAXX;
		// 		if ( is_worker ) {
		// 			for ( int i = 0; i != MAXX; ++i ) {
		// 				is_absent[ threadIdx.y ][ threadIdx.x ][ i ] = false;
		// 			}
		// 		}
		// 		wg_emit_cnt = 0;

		// 		__syncthreads();

		const auto cdu = 1.f / compMax( abs( pixel_in_out.ray.d ) );
		const auto opacity_threshold = 0.95f;
		const auto density = 1e-1f;

		auto pixel = pixel_in_out;
		auto &ray = pixel.ray;
		auto &nsteps = pixel.nsteps;

		while ( nsteps > 0 ) {
			vec3 ip = floor( ray.o );
			if ( int cd = chebyshev.sample_3d<int>( ip ) ) {
				nsteps -= skip_nblock_steps( ray, ip, cd, cdu, step );
			} else {
				auto present_id = present.sample_3d<int>( ip );
				if ( present_id != -1 ) {
					auto pt = ( cache_du.x + ( ray.o - ip ) ) * cache_du.y;
					auto spl = cache_tex[ present_id ].sample_3d<float>( pt );
					auto val = transfer_fn.sample_1d<vec4>( spl );
					// auto val = vec4( ray.o - ip, 1 );
					// auto val = vec4( chebyshev.sample_3d<float2>( ip ).x );
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

		// __syncthreads();

		// int wg_id = blockIdx.x + blockIdx.y * blockDim.x;
		// char *wg_base_ptr = absent_buf.ptr() + wg_len_bytes * wg_id;
		// int *wg_emit_cnt_ptr = (int *)wg_base_ptr;
		// uvec3 *wg_ptr = (uvec3 *)( wg_base_ptr + sizeof( int ) );

		// if ( is_worker ) {
		// 	for ( int i = 0; i != MAXX; ++i ) {
		// 		if ( is_absent[ threadIdx.y ][ threadIdx.x ][ i ] ) {
		// 			auto old = atomicAdd( &wg_emit_cnt, 1 );
		// 			if ( old < wg_max_emit_cnt ) {
		// 				wg_ptr[ old ] = absent_coord[ threadIdx.y ][ threadIdx.x ][ i ];
		// 			}
		// 		}
		// 	}
		// }

		// __syncthreads();

		// if ( threadIdx.x == 0 && threadIdx.y == 0 ) {
		// 	*wg_emit_cnt_ptr = min( wg_emit_cnt, wg_max_emit_cnt );
		// }
	}
};

REGISTER_SHADER_BUILDER(
  name( "volume_shader" )
	.cuda<VolumeShaderKernel>()
	.cpu<VolumeShaderKernel>(),
  VolumeShader );

VM_END_MODULE()
