#include <VMUtils/timer.hpp>
#include <varch/thumbnail.hpp>
#include <hydrant/double_buffering.hpp>
#include <hydrant/paging/rt_block_paging.hpp>
#include <varch/utils/io.hpp>
#include <hydrant/basic_renderer.hpp>
#include "paging_shader.hpp"

using namespace std;
using namespace vol;

struct PagingRenderer : BasicRenderer<PagingShader>
{
	using Super = BasicRenderer<PagingShader>;

	bool init( std::shared_ptr<Dataset> const &dataset,
			   RendererConfig const &cfg ) override;

	cufx::Image<> offline_render_ctxed( OfflineRenderCtx &ctx, Camera const &camera ) override;

	void realtime_render_dynamic( IRenderLoop &loop ) override;

private:
	vol::MtArchive *lvl0_arch;

	std::shared_ptr<vol::Thumbnail<int>> chebyshev_thumb;
	ThumbnailTexture<int> chebyshev;
};

bool PagingRenderer::init( std::shared_ptr<Dataset> const &dataset,
						   RendererConfig const &cfg )
{
	if ( !Super::init( dataset, cfg ) ) { return false; }

	auto params = cfg.params.get<PagingRendererConfig>();

	lvl0_arch = &dataset->meta.sample_levels[ 0 ].archives[ 0 ];

	chebyshev_thumb.reset(
	  new vol::Thumbnail<int>(
		dataset->root.resolve( lvl0_arch->thumbnails[ "chebyshev" ] ).resolved() ) );
	chebyshev = create_texture( chebyshev_thumb );
	shader.chebyshev = chebyshev.sampler();

	return true;
}

cufx::Image<> PagingRenderer::offline_render_ctxed( OfflineRenderCtx &ctx, Camera const &camera )
{
	auto film = create_film();
	return film.fetch_data().dump();
}

void PagingRenderer::realtime_render_dynamic( IRenderLoop &loop )
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

REGISTER_RENDERER( PagingRenderer, "Paging" );
}
