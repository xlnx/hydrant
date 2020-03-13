#include <VMUtils/timer.hpp>
#include <varch/thumbnail.hpp>
#include <hydrant/double_buffering.hpp>
#include <hydrant/paging/rt_block_paging.hpp>
#include <hydrant/paging/lossless_block_paging.hpp>
#include "isosurface.hpp"

using namespace std;
using namespace vol;

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	bool IsosurfaceRenderer::init( std::shared_ptr<Dataset> const &dataset,
								   RendererConfig const &cfg )
	{
		if ( !Super::init( dataset, cfg ) ) { return false; }

		vm::println( "STEP = {}", shader.step );
		vm::println( "MAX_STEPS = {}", shader.max_steps );
		vm::println( "MARCH_DIST = {}", shader.max_steps * shader.step );

		lvl0_arch = &dataset->meta.sample_levels[ 0 ].archives[ 0 ];

		chebyshev_thumb.reset(
		  new vol::Thumbnail<int>(
			dataset->root.resolve( lvl0_arch->thumbnails[ "chebyshev" ] ).resolved() ) );
		chebyshev = create_texture( chebyshev_thumb );
		shader.chebyshev = chebyshev.sampler();

		update( cfg.params );

		return true;
	}

	void IsosurfaceRenderer::update( vm::json::Any const &params_in )
	{
		Super::update( params_in );

		auto params = params_in.get<IsosurfaceRendererParams>();
		shader.mode = params.mode;
		shader.surface_color = params.surface_color;
		shader.isovalue = params.isovalue;
	}

	struct IsosurfaceRenderCtx : OfflineRenderCtx
	{
		mat4 et;
		unique_ptr<LosslessBlockPagingServer> srv;
		unique_ptr<OctreeCuller> culler;
	};

	OfflineRenderCtx *IsosurfaceRenderer::create_offline_render_ctx()
	{
		auto ctx = new IsosurfaceRenderCtx;
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

	cufx::Image<> IsosurfaceRenderer::offline_render_ctxed( OfflineRenderCtx & ctx_in, Camera const &camera )
	{
		auto &ctx = static_cast<IsosurfaceRenderCtx &>( ctx_in );
		auto opts = RaycastingOptions{}
					  .set_device( device );

		auto film = create_film();

		{
			auto state = ctx.srv->start( *ctx.culler, camera, ctx.et );
			bool emit = true;

			shader.to_world = inverse( exhibit.get_iet() );
			shader.light_pos = camera.position +
							   camera.target +
							   camera.up +
							   cross( camera.target, camera.up );
			shader.eye_pos = camera.position;

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
							  opts,
							  clear_color );

		return frame.fetch_data();
	}

	void IsosurfaceRenderer::realtime_render_dynamic( IRenderLoop & loop )
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

			  shader.to_world = inverse( exhibit.get_iet() );
			  shader.light_pos = loop.camera.position +
								 loop.camera.target +
								 loop.camera.up +
								 cross( loop.camera.target, loop.camera.up );
			  shader.eye_pos = loop.camera.position;
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
										opts,
										clear_color );
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

	REGISTER_RENDERER( IsosurfaceRenderer, "Isosurface" );
}

VM_END_MODULE()
