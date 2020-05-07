#include <VMUtils/timer.hpp>
#include <varch/utils/io.hpp>
#include <hydrant/dbuf_renderer.hpp>
#include <hydrant/paging/rt_block_paging.hpp>
#include <hydrant/paging/lossless_block_paging.hpp>
#include "isosurface_shader.hpp"

using namespace std;
using namespace vol;

struct IsosurfaceRenderer : DbufRenderer<IsosurfaceShader>
{
	using Super = DbufRenderer<IsosurfaceShader>;

	bool init( std::shared_ptr<Dataset> const &dataset,
			   RendererConfig const &cfg ) override;

	void update( vm::json::Any const &params ) override;

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
	vol::MtArchive *lvl0_arch;

	ThumbnailTexture<int> chebyshev;
};

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

cufx::Image<> IsosurfaceRenderer::offline_render_ctxed( OfflineRenderCtx &ctx_in, Camera const &camera )
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
	frame.update_device_view();

	return frame.fetch_data();
}

struct IsosurfaceRtRenderCtx : DbufRtRenderCtx
{
	Image<IsosurfaceShader::Pixel> film;
	Image<IsosurfaceFetchPixel> local;
	Image<IsosurfaceFetchPixel> recv;
	std::unique_ptr<RtBlockPagingServer> srv;

public:
	~IsosurfaceRtRenderCtx()
	{
		srv->stop();
	}
};

DbufRtRenderCtx *IsosurfaceRenderer::create_dbuf_rt_render_ctx()
{
	auto ctx = new IsosurfaceRtRenderCtx;
	ctx->film = create_film();
	ctx->local = Image<IsosurfaceFetchPixel>( ImageOptions{}
              	                              .set_device( device )
		                                      .set_resolution( resolution ) );
	ctx->recv = Image<IsosurfaceFetchPixel>( ImageOptions{}
                                             .set_resolution( resolution ) );
	vm::println( "device = {}", device.value().id() );
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

void IsosurfaceRenderer::dbuf_rt_render_frame( Image<cufx::StdByte3Pixel> &frame,
											   DbufRtRenderCtx &ctx_in,
											   IRenderLoop &loop,
											   OctreeCuller &culler,
											   MpiComm const &comm )
{
	auto &ctx = static_cast<IsosurfaceRtRenderCtx &>( ctx_in );
	
	std::size_t ns0, ns1, ns2;

	shader.to_world = inverse( exhibit.get_iet() );
	shader.light_pos = loop.camera.position +
		loop.camera.target +
		loop.camera.up +
		cross( loop.camera.target, loop.camera.up );
	shader.eye_pos = loop.camera.position;
	shader.paging = ctx.srv->update( culler, loop.camera );
	
	auto opts = RaycastingOptions{}.set_device( device );
	{
		vm::Timer::Scoped timer( [&]( auto dt ) {
				ns0 = dt.ns().cnt();
			} );

		raycaster.ray_emit_pass( exhibit,
								 loop.camera,
								 ctx.film.view(),
								 shader,
								 opts );
		raycaster.fetch_pass( ctx.film.view(),
							  ctx.local.view(),
							  shader,
							  opts );
		ctx.local.update_device_view();
	}

	{
		vm::Timer::Scoped timer( [&]( auto dt ) {
				ns1 = dt.ns().cnt();
			} );
		ctx.local.fetch_data();
	}

	vm::Timer::Scoped( [&]( auto dt ) {
			ns2 = dt.ns().cnt();
			ns0 /= ns2;
			ns1 /= ns2;
			//			vm::println("render/fetch/merge = {}/{}/{}", ns0, ns1, 1 );
		});
	
	int shl = 0;
	int flag = comm.size - 1;
	int rank = comm.rank;
	while ( flag ) {
		// sender
		if ( rank & 1 ) {
			auto dst = ( rank & -2 ) << shl;
			MPI_Send( &ctx.local.view().at_host( 0, 0 ), ctx.local.bytes(),
					  MPI_CHAR, dst, 0, comm.comm );
			break;
		} else if ( flag != rank ) {
			auto src = ( rank | 1 ) << shl;
			MPI_Recv( &ctx.recv.view().at_host( 0, 0 ), ctx.recv.bytes(),
					  MPI_CHAR, src, 0, comm.comm, MPI_STATUS_IGNORE );
			auto local_view = ctx.local.view();
			auto recv_view = ctx.recv.view();
			for ( int j = 0; j != local_view.height(); ++j ) {
				for ( int i = 0; i != local_view.width(); ++i ) {
					auto &local = local_view.at_host( i, j );
					auto &recv = recv_view.at_host( i, j );
					if ( recv.depth < local.depth ) {
						local.val = uchar3{ recv.val.x, 0, 0 };
					}
				}
			}
		}
		shl += 1;
		flag >>= 1;
		rank >>= 1;
	}
	
	if ( comm.rank == 0 ) {
		auto local_view = ctx.local.view();
		auto frame_view = frame.view();
		for ( int j = 0; j != local_view.height(); ++j ) {
			for ( int i = 0; i != local_view.width(); ++i ) {
				reinterpret_cast<uchar3&>( frame_view.at_host( i, j ) ) =
					local_view.at_host( i, j ).val;
			}
		}
	}
	
	MPI_Barrier( comm.comm );
}

REGISTER_RENDERER( IsosurfaceRenderer, "Isosurface" );
