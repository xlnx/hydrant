#include <VMUtils/timer.hpp>
#include <hydrant/dbuf_renderer.hpp>
#include <hydrant/paging/rt_block_paging.hpp>
#include <hydrant/paging/lossless_block_paging.hpp>
#include <hydrant/transfer_fn.hpp>
#include "volume_shader.hpp"

using namespace std;
using namespace vol;

struct VolumeRenderer : DbufRenderer<VolumeShader>
{
	using Super = DbufRenderer<VolumeShader>;

	bool init( std::shared_ptr<Dataset> const &dataset,
			   RendererConfig const &cfg ) override;

	void update( vm::json::Any const &params_in ) override;

protected:
	OfflineRenderCtx *create_offline_render_ctx() override;

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
	TransferFn transfer_fn;
	vol::MtArchive *lvl0_arch;

	ThumbnailTexture<int> chebyshev;
};

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

cufx::Image<> VolumeRenderer::offline_render_ctxed( OfflineRenderCtx &ctx_in, Camera const &camera )
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

	frame.update_device_view();
	return frame.fetch_data();
}

struct VolumeRtRenderCtx : DbufRtRenderCtx
{
	Image<VolumeShader::Pixel> film;
	std::unique_ptr<RtBlockPagingServer> srv;

public:
	~VolumeRtRenderCtx()
	{
		srv->stop();
	}
};

DbufRtRenderCtx *VolumeRenderer::create_dbuf_rt_render_ctx()
{
	auto ctx = new VolumeRtRenderCtx;
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

void VolumeRenderer::dbuf_rt_render_frame( Image<cufx::StdByte3Pixel> &frame,
										   DbufRtRenderCtx &ctx_in,
										   IRenderLoop &loop,
										   OctreeCuller &culler,
										   MpiComm const &comm )
{
	auto &ctx = static_cast<VolumeRtRenderCtx &>( ctx_in );

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
							  opts );
	}
}

REGISTER_RENDERER( VolumeRenderer, "Volume" );
