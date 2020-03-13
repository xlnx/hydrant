#include <VMUtils/timer.hpp>
#include <varch/thumbnail.hpp>
#include <hydrant/double_buffering.hpp>
#include <hydrant/unarchiver.hpp>
#include <hydrant/paging/rt_block_paging.hpp>
#include <hydrant/paging/lossless_block_paging.hpp>
#include "volume.hpp"

using namespace std;
using namespace vol;

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	bool VolumeRenderer::init( std::shared_ptr<Dataset> const &dataset,
							   RendererConfig const &cfg )
	{
		if ( !Super::init( dataset, cfg ) ) { return false; }

		auto params = cfg.params.get<VolumeRendererParams>();
		// shader.render_mode = params.mode == "volume" ? BrmVolume : BrmSolid;
		shader.density = params.density;
		transfer_fn = TransferFn( params.transfer_fn, device );
		shader.transfer_fn = transfer_fn.sampler();

		lvl0_arch = &dataset->meta.sample_levels[ 0 ].archives[ 0 ];

		chebyshev_thumb.reset(
		  new vol::Thumbnail<int>(
			dataset->root.resolve( lvl0_arch->thumbnails[ "chebyshev" ] ).resolved() ) );
		chebyshev = create_texture( chebyshev_thumb );
		shader.chebyshev = chebyshev.sampler();

		return true;
	}

	void VolumeRenderer::update( vm::json::Any const &params_in )
	{
		Super::update( params_in );

		auto params = params_in.get<VolumeRendererParams>();
		// shader.mode = params.mode;
		// shader.surface_color = params.surface_color;
		shader.density = params.density;
	}

	struct VolumeOfflineRenderCtx : OfflineRenderCtx
	{
		mat4 et;
		unique_ptr<LosslessBlockPagingServer> srv;
		unique_ptr<OctreeCuller> culler;
	};

	OfflineRenderCtx *VolumeRenderer::create_offline_render_ctx()
	{
		auto ctx = new VolumeOfflineRenderCtx;
		ctx->et = inverse( exhibit.get_iet() );
		ctx->srv.reset( new LosslessBlockPagingServer(
		  LosslessBlockPagingServerOptions{}
			.set_dataset( dataset )
			.set_device( device )
			.set_storage_opts( cufx::Texture::Options{}
								 .set_address_mode( cufx::Texture::AddressMode::Wrap )
								 .set_filter_mode( cufx::Texture::FilterMode::Linear )
								 .set_read_mode( cufx::Texture::ReadMode::NormalizedFloat )
								 .set_normalize_coords( true ) ) ) );
		ctx->culler.reset( new OctreeCuller( exhibit, chebyshev_thumb ) );
		return ctx;
	}

	cufx::Image<> VolumeRenderer::offline_render_ctxed( OfflineRenderCtx & ctx_in, Camera const &camera )
	{
		auto &ctx = static_cast<VolumeOfflineRenderCtx &>( ctx_in );
		auto opts = RaycastingOptions{}
					  .set_device( device );

		auto film = create_film();

		{
			auto state = ctx.srv->start( *ctx.culler, camera, ctx.et );
			bool emit = true;

			while ( state.next( shader.paging ) ) {
				if ( emit ) {
					raycaster.ray_emit_pass( exhibit,
											 camera,
											 film.view(),
											 shader,
											 opts );
					emit = false;
				} else {
					raycaster.ray_march_pass( film.view(),
											  shader,
											  opts );
				}
			}
		}

		auto frame = Image<cufx::StdByte3Pixel>( ImageOptions{}
												   .set_device( device )
												   .set_resolution( resolution ) );

		raycaster.pixel_pass( film.view(),
							  frame.view(),
							  shader,
							  opts );

		return frame.fetch_data();
	}

	void VolumeRenderer::realtime_render_dynamic( IRenderLoop & loop )
	{
		auto film = create_film();

		auto opts = RtBlockPagingServerOptions{}
					  .set_dim( dim )
					  .set_dataset( dataset )
					  .set_device( device )
					  .set_storage_opts( cufx::Texture::Options{}
										   .set_address_mode( cufx::Texture::AddressMode::Wrap )
										   .set_filter_mode( cufx::Texture::FilterMode::Linear )
										   .set_read_mode( cufx::Texture::ReadMode::NormalizedFloat )
										   .set_normalize_coords( true ) );
		RtBlockPagingServer srv( opts );
		OctreeCuller culler( exhibit, chebyshev_thumb );

		FnDoubleBuffering loop_drv(
		  ImageOptions{}
			.set_device( device )
			.set_resolution( resolution ),
		  loop,
		  [&]( auto &frame, auto frame_idx ) {
			  std::size_t ns = 0, ns1 = 0;

			  shader.paging = srv.update( culler, loop.camera );

			  {
				  vm::Timer::Scoped timer( [&]( auto dt ) {
					  ns1 += dt.ns().cnt();
				  } );

				  auto opts = RaycastingOptions{}.set_device( device );

				  raycaster.ray_emit_pass( exhibit,
										   loop.camera,
										   film.view(),
										   shader,
										   opts );

				  raycaster.pixel_pass( film.view(),
										frame.view(),
										shader,
										opts );
			  }
		  },
		  [&]( auto &frame, auto frame_idx ) {
			  auto fp = frame.fetch_data();
			  loop.on_frame( fp );
		  } );

		srv.start();
		loop_drv.run();
		srv.stop();
	}

	REGISTER_RENDERER( VolumeRenderer, "Volume" );
}

VM_END_MODULE()
