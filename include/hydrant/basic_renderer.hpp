#pragma once

#include <glog/logging.h>
#include <varch/thumbnail.hpp>
#include <cudafx/device.hpp>
#include <hydrant/bridge/image.hpp>
#include <hydrant/core/renderer.hpp>
#include <hydrant/bridge/texture_3d.hpp>
#include <hydrant/core/render_loop.hpp>
#include <hydrant/basic_renderer.schema.hpp>

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	template <typename Shader>
	struct BasicRenderer;

	template <typename T>
	struct ThumbnailTexture : Texture3D<T>
	{
		using Texture3D<T>::Texture3D;

	private:
		std::shared_ptr<vol::Thumbnail<T>> thumb;
		template <typename Shader>
		friend struct BasicRenderer;
	};

	struct OfflineRenderCtx : vm::Dynamic
	{
	};

	template <typename Shader>
	struct BasicRenderer : IRenderer
	{
		bool init( std::shared_ptr<Dataset> const &dataset,
				   RendererConfig const &cfg ) override
		{
			if ( !IRenderer::init( dataset, cfg ) ) { return false; }

			auto params = cfg.params.get<BasicRendererParams>();
			if ( params.device == ShadingDevice::Cuda ) {
				auto devices = cufx::Device::scan();
				if ( !devices.size() ) {
					LOG( ERROR ) << vm::fmt( "cuda device not found, fallback to cpu render mode" );
				} else {
					if ( params.comm_rank >= devices.size() ) {
						LOG( FATAL ) << vm::fmt( "comm_rank >= devices.size()" );
					} else {
						device = devices[ params.comm_rank ];
						lk = device.value().lock();
					}
				}
			}

			auto &lvl0 = dataset->meta.sample_levels[ 0 ];
			auto blksz = float( dataset->meta.block_size );
			dim = vec3( lvl0.dim.x,
						lvl0.dim.y,
						lvl0.dim.z );
			auto raw = vec3( lvl0.raw.x,
							 lvl0.raw.y,
							 lvl0.raw.z );
			auto f_dim = raw / blksz;
			exhibit = Exhibit{}
						.set_center( f_dim / 2.f )
						.set_size( f_dim );

			shader.bbox = Box3D{ { 0, 0, 0 }, f_dim };
			shader.du = 1.f / blksz;

			update( cfg.params );
			
			return true;
		}

		void update( vm::json::Any const &params_in ) override
		{
			auto params = params_in.get<BasicRendererParams>();
			shader.max_steps = params.max_steps;
			shader.step = shader.du / 4.f / params.sample_rate;
			clear_color = params.clear_color;
		}

	public:
		cufx::Image<> offline_render( Camera const &camera ) override final
		{
			vm::Option<cufx::Device::Lock> lk;
			if ( device.has_value() ) {
				lk = device.value().lock();
			}
			std::unique_ptr<OfflineRenderCtx> pctx( create_offline_render_ctx() );
			return offline_render_ctxed( *pctx, camera );
		}

	protected:
		virtual cufx::Image<> offline_render_ctxed( OfflineRenderCtx &ctx, Camera const &camera ) = 0;

		virtual OfflineRenderCtx *create_offline_render_ctx()
		{
			return new OfflineRenderCtx;
		}

	public:
		void realtime_render( IRenderLoop &loop, RealtimeRenderOptions const &opts ) override final
		{
			vm::Option<cufx::Device::Lock> lk;
			if ( device.has_value() ) {
				lk = device.value().lock();
			}
			switch ( opts.quality._to_integral() ) {
			case RealtimeRenderQuality::Lossless: {
				realtime_render_lossless( loop, opts.comm );
			} break;
			case RealtimeRenderQuality::Dynamic: {
				realtime_render_dynamic( loop, opts.comm );
			} break;
			}
		}

	protected:
		void realtime_render_default( IRenderLoop &loop, MpiComm const &comm )
		{
			std::unique_ptr<OfflineRenderCtx> pctx( create_offline_render_ctx() );
			loop.post_loop();
			while ( !loop.should_stop() ) {
				loop.post_frame();
				auto frame = offline_render_ctxed( *pctx, loop.camera );
				loop.on_frame( frame );
				loop.after_frame();
			}
			loop.after_loop();
		}

		virtual void realtime_render_lossless( IRenderLoop &loop, MpiComm const &comm )
		{
			realtime_render_default( loop, comm );
		}

		virtual void realtime_render_dynamic( IRenderLoop &loop, MpiComm const &comm )
		{
			realtime_render_default( loop, comm );
		}

	public:
		template <typename T>
		ThumbnailTexture<T> create_texture( std::shared_ptr<vol::Thumbnail<T>> const &thumb ) const
		{
			auto opts = Texture3DOptions{}
						  .set_device( device )
						  .set_dim( thumb->dim.x, thumb->dim.y, thumb->dim.z )
						  .set_opts( cufx::Texture::Options::as_array()
									   .set_address_mode( cufx::Texture::AddressMode::Clamp ) );
			ThumbnailTexture<T> texture( opts );
			if ( !device.has_value() ) { texture.thumb = thumb; }
			texture.source( thumb->data() );
			return texture;
		}

		Image<typename Shader::Pixel> create_film() const
		{
			auto img_opts = ImageOptions{}
							  .set_device( device )
							  .set_resolution( resolution );
			return Image<typename Shader::Pixel>( img_opts );
		}

	protected:
		vm::Option<cufx::Device> device;
		vec3 clear_color;
		uvec3 dim;
		Shader shader;
		Exhibit exhibit;
	private:
		vm::Option<cufx::Device::Lock> lk;
	};
}

VM_END_MODULE()
