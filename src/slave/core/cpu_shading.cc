#include <thread>
#include <vector>
#include <hydrant/core/shader.hpp>

VM_BEGIN_MODULE( hydrant )

using namespace std;

void ray_emit_task_dispatch( ThreadPoolInfo const &thread_pool_info,
							 CpuRayEmitKernelArgs const &args )
{
	vector<thread> threads;
	for ( int i = 0; i < thread_pool_info.nthreads; ++i ) {
		threads.emplace_back(
		  [&, y0 = i] {
			  auto cc = vec2( args.image_desc.resolution ) / 2.f;
			  auto launcher = (ray_emit_shader_t *)args.launcher;
			  Ray ray = { args.view.ray_o, {} };
			  for ( int y = y0; y < args.image_desc.resolution.y; y += thread_pool_info.nthreads ) {
				  for ( int x = 0; x < args.image_desc.resolution.x; ++x ) {
					  auto uv = ( vec2{ x, y } - cc ) * 2.f / float( args.image_desc.resolution.y );
					  ray.d = normalize( vec3( args.view.trans * vec4( uv.x, -uv.y, -args.view.ctg_fovy_2, 1 ) ) - args.view.ray_o );
					  launcher( ray,
								args.image_desc.data + args.image_desc.pixel_size * ( args.image_desc.resolution.x * y + x ),
								args.shader );
				  }
			  }
		  } );
	}
	for ( auto &t : threads ) { t.join(); }
}

void ray_march_task_dispatch( ThreadPoolInfo const &thread_pool_info,
							  CpuRayMarchKernelArgs const &args )
{
	vector<thread> threads;
	for ( int i = 0; i < thread_pool_info.nthreads; ++i ) {
		threads.emplace_back(
		  [&, y0 = i] {
			  auto launcher = (ray_march_shader_t *)args.launcher;
			  for ( int y = y0; y < args.image_desc.resolution.y; y += thread_pool_info.nthreads ) {
				  for ( int x = 0; x < args.image_desc.resolution.x; ++x ) {
					  launcher( args.image_desc.data + args.image_desc.pixel_size * ( args.image_desc.resolution.x * y + x ),
								args.shader );
				  }
			  }
		  } );
	}
	for ( auto &t : threads ) { t.join(); }
}

void pixel_task_dispatch( ThreadPoolInfo const &thread_pool_info,
						  CpuPixelKernelArgs const &args )
{
	vector<thread> threads;
	for ( int i = 0; i < thread_pool_info.nthreads; ++i ) {
		threads.emplace_back(
		  [&, y0 = i] {
			  auto launcher = (pixel_shader_t *)args.launcher;
			  for ( int y = y0; y < args.image_desc.resolution.y; y += thread_pool_info.nthreads ) {
				  for ( int x = 0; x < args.image_desc.resolution.x; ++x ) {
					  launcher( args.image_desc.data + args.image_desc.pixel_size * ( args.image_desc.resolution.x * y + x ),
								args.dst_desc.data + args.dst_desc.pixel_size * ( args.dst_desc.resolution.x * y + x ),
								&args.clear_color );
				  }
			  }
		  } );
	}
	for ( auto &t : threads ) { t.join(); }
}

void fetch_task_dispatch( ThreadPoolInfo const &thread_pool_info,
						  CpuFetchKernelArgs const &args )
{
	vector<thread> threads;
	for ( int i = 0; i < thread_pool_info.nthreads; ++i ) {
		threads.emplace_back(
		  [&, y0 = i] {
			  auto launcher = (fetch_shader_t *)args.launcher;
			  for ( int y = y0; y < args.image_desc.resolution.y; y += thread_pool_info.nthreads ) {
				  for ( int x = 0; x < args.image_desc.resolution.x; ++x ) {
					  launcher( args.image_desc.data + args.image_desc.pixel_size * ( args.image_desc.resolution.x * y + x ),
								args.dst_desc.data + args.dst_desc.pixel_size * ( args.dst_desc.resolution.x * y + x ),
								args.shader );
				  }
			  }
		  } );
	}
	for ( auto &t : threads ) { t.join(); }
}

VM_END_MODULE()
