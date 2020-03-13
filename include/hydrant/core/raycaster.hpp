#pragma once

#include <thread>
#include <fstream>
#include <algorithm>
#include <glog/logging.h>
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
	struct RaycastingOptions
	{
		VM_DEFINE_ATTRIBUTE( vm::Option<cufx::Device>, device );
	};

	struct Raycaster
	{
		static int round_up_div( int a, int b )
		{
			return a % b == 0 ? a / b : a / b + 1;
		}

		template <typename P, typename F>
		void ray_emit_pass( Exhibit const &e,
							Camera const &c,
							cufx::ImageView<P> &img,
							F const &f,
							RaycastingOptions const &opts )
		{
			if ( opts.device.has_value() ) {
				CudaRayEmitKernelArgs kernel_args;
				kernel_args.shading_pass = ShadingPass::RayEmit;
				kernel_args.image_desc.create_from_img( img, true );
				fill_ray_emit_args( kernel_args, e, c );

				return cast_cuda_impl( &kernel_args, f, opts );
			} else {
				CpuRayEmitKernelArgs kernel_args;
				kernel_args.shading_pass = ShadingPass::RayEmit;
				kernel_args.image_desc.create_from_img( img, false );
				fill_ray_emit_args( kernel_args, e, c );

				return cast_cpu_impl( &kernel_args, f, opts );
			}
		}

		template <typename P, typename F>
		void ray_march_pass( cufx::ImageView<P> &img,
							 F const &f,
							 RaycastingOptions const &opts )
		{
			if ( opts.device.has_value() ) {
				CudaRayMarchKernelArgs kernel_args;
				kernel_args.shading_pass = ShadingPass::RayMarch;
				kernel_args.image_desc.create_from_img( img, true );

				return cast_cuda_impl( &kernel_args, f, opts );
			} else {
				CpuRayMarchKernelArgs kernel_args;
				kernel_args.shading_pass = ShadingPass::RayMarch;
				kernel_args.image_desc.create_from_img( img, false );

				return cast_cpu_impl( &kernel_args, f, opts );
			}
		}

		template <typename P, typename F>
		void pixel_pass( cufx::ImageView<P> &img,
						 cufx::ImageView<cufx::StdByte3Pixel> &dst,
						 F const &f,
						 RaycastingOptions const &opts,
						 vec3 const &clear_color = vec3( 0 ) )
		{
			if ( opts.device.has_value() ) {
				CudaPixelKernelArgs kernel_args;
				kernel_args.shading_pass = ShadingPass::Pixel;
				kernel_args.image_desc.create_from_img( img, true );
				kernel_args.dst_desc.create_from_img( dst, true );
				kernel_args.clear_color = saturate( clear_color );

				return cast_cuda_impl( &kernel_args, f, opts );
			} else {
				CpuPixelKernelArgs kernel_args;
				kernel_args.shading_pass = ShadingPass::Pixel;
				kernel_args.image_desc.create_from_img( img, false );
				kernel_args.dst_desc.create_from_img( dst, false );
				kernel_args.clear_color = saturate( clear_color );

				return cast_cpu_impl( &kernel_args, f, opts );
			}
		}

	private:
		void fill_ray_emit_args( BasicRayEmitKernelArgs &args,
								 Exhibit const &e,
								 Camera const &c ) const
		{
			auto iet = e.get_iet();
			auto itrans = iet * c.get_ivt();
			args.view.trans = itrans;
			args.view.ctg_fovy_2 = c.ctg_fovy_2;
			args.view.ray_o = iet * vec4{ c.position.x, c.position.y, c.position.z, 1 };
		}

		template <typename F>
		std::function<void( void * )> get_shading_device( ShadingDevice::_enumerated dev ) const
		{
			auto shader_meta = ShaderRegistry::instance().meta.find( typeid( F ) );
			if ( shader_meta == ShaderRegistry::instance().meta.end() ) {
				LOG( FATAL ) << "no such shader";
			}

			auto cuda_shader_kernel = shader_meta->second.devices.find( dev );
			if ( cuda_shader_kernel == shader_meta->second.devices.end() ) {
				LOG( FATAL ) << "no cuda shader kernel";
			}

			return cuda_shader_kernel->second;
		}

		template <typename F>
		void cast_cuda_impl( BasicKernelArgs *kernel_args,
							 F const &f,
							 RaycastingOptions const &opts )
		{
			CudaShadingArgs args;
			args.kernel_args = kernel_args;
			args.shader = &f;

			auto kernel_block_dim = dim3( 32, 32 );
			args.launch_info = cufx::KernelLaunchInfo{}
								 .set_device( opts.device.value() )
								 .set_grid_dim( round_up_div( kernel_args->image_desc.resolution.x,
															  kernel_block_dim.x ),
												round_up_div( kernel_args->image_desc.resolution.y,
															  kernel_block_dim.y ) )
								 .set_block_dim( kernel_block_dim );

			auto device = get_shading_device<F>( ShadingDevice::Cuda );
			device( reinterpret_cast<void *>( &args ) );
		}

		template <typename F>
		void cast_cpu_impl( BasicKernelArgs *kernel_args,
							F const &f,
							RaycastingOptions const &opts )
		{
			CpuShadingArgs args;
			args.kernel_args = kernel_args;
			args.shader = &f;
			args.thread_pool_info.nthreads = std::thread::hardware_concurrency();

			auto device = get_shading_device<F>( ShadingDevice::Cpu );
			device( reinterpret_cast<void *>( &args ) );
		}
	};
}

VM_END_MODULE()
