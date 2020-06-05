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
	
	std::size_t dbuf_rt_render_frame( Image<cufx::StdByte3Pixel> &frame,
							   DbufRtRenderCtx &ctx,
							   IRenderLoop &loop,
							   OctreeCuller &culler,
							   MpiComm const &comm,
							   std::vector<int> const &z_order ) override;

private:
	std::size_t mem_limit_mb;
	ThumbnailTexture<int> chebyshev;
};

bool IsosurfaceRenderer::init( std::shared_ptr<Dataset> const &dataset,
							   RendererConfig const &cfg )
{
	if ( !Super::init( dataset, cfg ) ) { return false; }

	vm::println( "STEP = {}", shader.step );
	vm::println( "MAX_STEPS = {}", shader.max_steps );
	vm::println( "MARCH_DIST = {}", shader.max_steps * shader.step );

	chebyshev_thumb.reset(
	  new vol::Thumbnail<int>(
		dataset->root.resolve( dataset->meta.sample_levels[ 0 ].thumbnails[ "chebyshev" ] ).resolved() ) );
	chebyshev = create_texture( chebyshev_thumb );
	shader.chebyshev = chebyshev.sampler();

	update( cfg.params );

	return true;
}

void IsosurfaceRenderer::update( vm::json::Any const &params_in )
{
	Super::update( params_in );

	auto params = params_in.get<IsosurfaceRendererParams>();
	mem_limit_mb = params.mem_limit_mb;
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
	std::unique_ptr<Image<cufx::StdByte3Pixel>> tmp;
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
	auto opts = RtBlockPagingServerOptions{}
				  .set_dim( dim )
				  .set_dataset( dataset )
				  .set_device( device )
				  .set_mem_limit_mb( mem_limit_mb )
				  .set_storage_opts( cufx::Texture::Options{}
									   .set_address_mode( cufx::Texture::AddressMode::Wrap )
									   .set_filter_mode( cufx::Texture::FilterMode::Linear )
									   .set_read_mode( cufx::Texture::ReadMode::NormalizedFloat )
									   .set_normalize_coords( true ) );
	ctx->srv.reset( new RtBlockPagingServer( opts ) );
	ctx->srv->start();
	return ctx;
}

std::size_t IsosurfaceRenderer::dbuf_rt_render_frame( Image<cufx::StdByte3Pixel> &frame,
											   DbufRtRenderCtx &ctx_in,
											   IRenderLoop &loop,
											   OctreeCuller &culler,
											   MpiComm const &comm,
											   std::vector<int> const &z_order )
{
	auto &ctx = static_cast<IsosurfaceRtRenderCtx &>( ctx_in );

	std::size_t render_t;
	
	std::size_t ns0, ns1, ns2;

	shader.to_world = inverse( exhibit.get_iet() );
	shader.light_pos = loop.camera.position +
		loop.camera.target +
		loop.camera.up +
		cross( loop.camera.target, loop.camera.up );
	shader.eye_pos = loop.camera.position;
	shader.paging = ctx.srv->update( culler, loop.camera );
	
	if ( !ctx.tmp ) {
		ctx.tmp.reset( new Image<cufx::StdByte3Pixel>( ImageOptions{}
		                                    .set_resolution( resolution.x,
															 resolution.y / comm.size ) ) );
	}

	auto opts = RaycastingOptions{}.set_device( device );
	{
		vm::Timer::Scoped timer( [&]( auto dt ) {
				render_t = dt.ns().cnt();
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

	vm::Timer::Scoped timer( [&]( auto dt ) {
			ns2 = dt.ns().cnt();
			auto m = std::min( ns0, std::min( ns1, ns2 ) );
			ns0 /= m;
			ns1 /= m;
			ns2 /= m;
			if ( comm.rank == 0 ) {
				// vm::println("render/fetch/merge = {}/{}/{}", ns0, ns1, ns2 );
			}
		});

	auto local_view = ctx.local.view();
	auto recv_view = ctx.recv.view();
	auto s_height = ctx.recv.view().height() / comm.size;
	auto s_bytes = ctx.recv.bytes() / recv_view.height() * s_height;

	std::vector<MPI_Request> rs( comm.size * 2 );
	for ( int i = 0; i < comm.size; ++i ) {
		if ( i != comm.rank ) {
			MPI_Isend( &local_view.at_host( 0, i * s_height ), s_bytes,
					   MPI_CHAR, i, 0, comm.comm, &rs[ i ] );
			MPI_Irecv( &recv_view.at_host( 0, i * s_height ), s_bytes,
					   MPI_CHAR, i, 0, comm.comm, &rs[ i + comm.size ] );
		} else {
			memcpy( &recv_view.at_host( 0, i * s_height ),
					&local_view.at_host( 0, i * s_height ), s_bytes );
		}
	}
	for ( int i = 0; i < comm.size; ++i ) {
		if ( i != comm.rank ) {
			MPI_Wait( &rs[ i ], MPI_STATUS_IGNORE );
			MPI_Wait( &rs[ i + comm.size ], MPI_STATUS_IGNORE );
		}
	}
	
	auto f0 = z_order[ 0 ] * s_height;
	for ( int k = 1; k < comm.size; ++k ) {
		auto f1 = z_order[ k ] * s_height;
		for ( int j = 0; j < s_height; ++j ) {
			for ( int i = 0; i < recv_view.width(); ++i ) {
				auto &front = recv_view.at_host( i, j + f0 );
				auto &back = recv_view.at_host( i, j + f1 );
				if ( back.depth < front.depth ) { front = back; }
			}
		}
	}
	auto tmp_view = ctx.tmp->view();
	for ( int j = 0; j < s_height; ++j ) {
		for ( int i = 0; i < recv_view.width(); ++i ) {
			auto &val = recv_view.at_host( i, j + f0 ).val;
			reinterpret_cast<uchar3&>( tmp_view.at_host( i, j ) ) =
				uchar3{ val.x, val.y, val.z };
		}
	}

	auto frame_view = frame.view();
	auto t_bytes = ctx.tmp->bytes();
	for ( int i = 1; i < comm.size; ++i ) {
		if ( comm.rank == 0 ) {
			MPI_Recv( &frame_view.at_host( 0, i * s_height ), t_bytes,
					  MPI_CHAR, i, 0, comm.comm, MPI_STATUS_IGNORE );
		} else if ( i == comm.rank ) {
			MPI_Send( &tmp_view.at_host( 0, 0 ), t_bytes,
					  MPI_CHAR, 0, 0, comm.comm );
		}
	}
	if ( comm.rank == 0 ) {
		memcpy( &frame_view.at_host( 0, 0 ),
				&tmp_view.at_host( 0, 0 ), t_bytes );
	}

	MPI_Barrier( comm.comm );

	return render_t;
}

REGISTER_RENDERER( IsosurfaceRenderer, "Isosurface" );
