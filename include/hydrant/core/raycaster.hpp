#pragma once

#include <thread>
#include <fstream>
#include <algorithm>
#include <cudafx/device.hpp>
#include <cudafx/kernel.hpp>
#include <cudafx/image.hpp>
#include <hydrant/core/glm_math.hpp>
#include <hydrant/core/scene.hpp>
#include <hydrant/core/shader.hpp>

VM_BEGIN_MODULE( hydrant )

/* Raycaster */

VM_EXPORT
{
	struct Raycaster
	{
		static int round_up_div( int a, int b )
		{
			return a % b == 0 ? a / b : a / b + 1;
		}
		template <typename P, typename F>
		void cast( Exhibit const &e, Camera const &c, cufx::ImageView<P> &img, F const &f )
		{
			auto et = e.get_matrix();
			auto ivt = inverse( c.get_matrix() );

			CudaRayEmitShadingKernelArgs kernel_args;
			kernel_args.shading_pass = ShadingPass::RayEmit;
			kernel_args.image_desc.create_from_img( img );
			kernel_args.view.trans = et * ivt;
			kernel_args.view.itg_fovy = 1. / tan( M_PI / 3 / 2 );
			kernel_args.view.ray_o = et * vec4{ c.position.x, c.position.y, c.position.z, 1 };

			return cast_impl( &kernel_args, img, f );
		}
		template <typename P, typename F>
		void cast( cufx::ImageView<P> &img, F const &f )
		{
			CudaPixelShadingKernelArgs kernel_args;
			kernel_args.shading_pass = ShadingPass::Pixel;
			kernel_args.image_desc.create_from_img( img );

			return cast_impl( &kernel_args, img, f );
		}

	private:
		template <typename P, typename F>
		void cast_impl( BasicCudaShadingKernelArgs *kernel_args, cufx::ImageView<P> &img, F const &f )
		{
			CudaShadingArgs args;
			args.kernel_args = kernel_args;
			args.shader = &f;

			auto kernel_block_dim = dim3( 32, 32 );
			args.launch_info = cufx::KernelLaunchInfo{}
								 .set_device( cufx::Device::scan()[ 0 ] )
								 .set_grid_dim( round_up_div( kernel_args->image_desc.resolution.x,
															  kernel_block_dim.x ),
												round_up_div( kernel_args->image_desc.resolution.y,
															  kernel_block_dim.y ) )
								 .set_block_dim( kernel_block_dim );

			auto shader_meta = ShaderRegistry::instance.meta.find( typeid( F ) );
			if ( shader_meta == ShaderRegistry::instance.meta.end() ) {
				throw std::logic_error( "no such shader" );
			}

			auto cuda_shader_kernel = shader_meta->second.devices.find( ShadingDevice::Cuda );
			if ( cuda_shader_kernel == shader_meta->second.devices.end() ) {
				throw std::logic_error( "no cuda shader kernel" );
			}

			cuda_shader_kernel->second( reinterpret_cast<void *>( &args ) );
		}

		// private:
		// 	template <typename P, typename F, bool IsGpuFn>
		// 	struct CastImpl;

		// 	template <typename P, typename F, bool IsGpuFn>
		// 	friend struct CastImpl;

		// 	template <typename P, typename F>
		// 	void cpu_cast_impl( RayEmitKernelArgs const &args, cufx::ImageView<P> &img, F const &f )
		// 	{
		// 		auto cc = vec2( args.image.resolution ) / 2.f;
		// 		auto nthreads = std::thread::hardware_concurrency();
		// 		std::vector<std::thread> threads;
		// 		for ( int i = 0; i != nthreads; ++i ) {
		// 			threads.emplace_back(
		// 			  [&, y0 = i] {
		// 				  Ray ray = { args.view.ray_o, {} };
		// 				  for ( int y = y0; y < args.image.resolution.y; y += nthreads ) {
		// 					  for ( int x = 0; x < args.image.resolution.x; ++x ) {
		// 						  auto uv = ( vec2{ x, y } - cc ) * 2.f / float( args.image.resolution.y );
		// 						  ray.d = normalize( vec3( args.view.trans * vec4( uv.x, -uv.y, -args.view.itg_fovy, 1 ) ) - ray.o );
		// 						  f( ray, img.at( x, y ) );
		// 					  }
		// 				  }
		// 			  } );
		// 		}
		// 		for ( auto &t : threads ) { t.join(); }
		// 	}
	};

	// template <typename P, typename F>
	// struct Raycaster::CastImpl<P, F, false>
	// {
	// 	static void apply( RayEmitKernelArgs &args, cufx::ImageView<P> &img, F const &f, Raycaster *self )
	// 	{
	// 		args.image.data = reinterpret_cast<char *>( &img.at_host( 0, 0 ) );
	// 		self->cpu_cast_impl( args, img, f );
	// 	}
	// };

	// template <typename P, typename F>
	// struct Raycaster::CastImpl<P, F, true>
	// {
	// 	static int round_up_div( int a, int b )
	// 	{
	// 		return a % b == 0 ? a / b : a / b + 1;
	// 	}
	// 	static void apply( RayEmitKernelArgs &args, cufx::ImageView<P> &img, F const &f, Raycaster *self )
	// 	{
	// 		static_assert( std::is_trivially_copyable<F>::value, "shader must be trivially copyable" );

	// 		auto kernel_block_dim = dim3( 32, 32 );
	// 		auto launch_info = cufx::KernelLaunchInfo{}
	// 							 .set_device( cufx::Device::scan()[ 0 ] )
	// 							 .set_grid_dim( round_up_div( args.image.resolution.x, kernel_block_dim.x ),
	// 											round_up_div( args.image.resolution.y, kernel_block_dim.y ) )
	// 							 .set_block_dim( kernel_block_dim );
	// 		ray_emit_kernel( launch_info, args ).launch();
	// 	}
	// };
}

VM_END_MODULE()
