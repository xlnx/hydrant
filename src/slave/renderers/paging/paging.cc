#include <VMUtils/timer.hpp>
#include <varch/utils/io.hpp>
#include <hydrant/dbuf_renderer.hpp>
#include <hydrant/paging/rt_block_paging.hpp>
#include "paging_shader.hpp"

using namespace std;
using namespace vol;

struct PagingRenderer : DbufRenderer<PagingShader>
{
	using Super = DbufRenderer<PagingShader>;

	bool init( std::shared_ptr<Dataset> const &dataset,
			   RendererConfig const &cfg ) override;

protected:
	cufx::Image<> offline_render_ctxed( OfflineRenderCtx &ctx,
										Camera const &camera ) override;

protected:
	DbufRtRenderCtx *create_dbuf_rt_render_ctx() override;
	
	void dbuf_rt_render_frame( Image<cufx::StdByte3Pixel> &frame,
							   DbufRtRenderCtx &ctx,
							   IRenderLoop &loop,
							   OctreeCuller &culler,
							   MpiComm const &comm ) override;

private:
	vol::MtArchive *lvl0_arch;

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
	film.update_device_view();
	return film.fetch_data().dump();
}

struct PagingRtRenderCtx : DbufRtRenderCtx
{
	Image<PagingShader::Pixel> film;
	std::unique_ptr<RtBlockPagingServer> srv;

public:
	~PagingRtRenderCtx()
	{
		srv->stop();
	}
};

DbufRtRenderCtx *PagingRenderer::create_dbuf_rt_render_ctx()
{
	auto ctx = new PagingRtRenderCtx;
	ctx->film = create_film();
	auto opts = RtBlockPagingServerOptions{}
				  .set_dim( dim )
				  .set_dataset( dataset )
				  .set_device( device )
				  .set_storage_opts( cufx::Texture::Options{}
									   .set_address_mode( cufx::Texture::AddressMode::Wrap )
									   .set_filter_mode( cufx::Texture::FilterMode::Linear )
									   .set_read_mode( cufx::Texture::ReadMode::NormalizedFloat )
									   .set_normalize_coords( true ) );
	ctx->srv.reset( new RtBlockPagingServer( opts ) );
	ctx->srv->start();
	return ctx;
}

void PagingRenderer::dbuf_rt_render_frame( Image<cufx::StdByte3Pixel> &frame,
										   DbufRtRenderCtx &ctx_in,
										   IRenderLoop &loop,
										   OctreeCuller &culler,
										   MpiComm const &comm )
{
	auto &ctx = static_cast<PagingRtRenderCtx &>( ctx_in );

	std::size_t ns = 0, ns1 = 0;
	
	shader.paging = ctx.srv->update( culler, loop.camera );
	
	{
		vm::Timer::Scoped timer( [&]( auto dt ) {
				ns1 += dt.ns().cnt();
			} );
		
		auto opts = RaycastingOptions{}.set_device( device );
		
		raycaster.ray_emit_pass( exhibit,
								 loop.camera,
								 ctx.film.view(),
								 shader,
								 opts );
		
		raycaster.pixel_pass( ctx.film.view(),
							  frame.view(),
							  shader,
							  opts,
							  clear_color );
	}
}

REGISTER_RENDERER( PagingRenderer, "Paging" );
