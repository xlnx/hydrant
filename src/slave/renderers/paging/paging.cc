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
	
	std::size_t dbuf_rt_render_frame( Image<cufx::StdByte3Pixel> &frame,
							   DbufRtRenderCtx &ctx,
							   IRenderLoop &loop,
							   OctreeCuller &culler,
							   MpiComm const &comm,
							   std::vector<int> const &z_order ) override;

private:
	ThumbnailTexture<int> chebyshev;
};

bool PagingRenderer::init( std::shared_ptr<Dataset> const &dataset,
						   RendererConfig const &cfg )
{
	if ( !Super::init( dataset, cfg ) ) { return false; }

	auto params = cfg.params.get<PagingRendererConfig>();

	chebyshev_thumb.reset(
	  new vol::Thumbnail<int>(
		dataset->root.resolve( dataset->meta.sample_levels[ 0 ].thumbnails[ "chebyshev" ] ).resolved() ) );
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
	Image<PagingFetchPixel> local;
	Image<PagingFetchPixel> recv;
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
	ctx->local = Image<PagingFetchPixel>( ImageOptions{}
              	                              .set_device( device )
		                                      .set_resolution( resolution ) );
	ctx->recv = Image<PagingFetchPixel>( ImageOptions{}
                                             .set_resolution( resolution ) );
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

std::size_t PagingRenderer::dbuf_rt_render_frame( Image<cufx::StdByte3Pixel> &frame,
										   DbufRtRenderCtx &ctx_in,
										   IRenderLoop &loop,
										   OctreeCuller &culler,
										   MpiComm const &comm,
										   std::vector<int> const &z_order )
{
	auto &ctx = static_cast<PagingRtRenderCtx &>( ctx_in );

	std::size_t render_t;
	
	std::size_t ns0, ns1, ns2;
	
	shader.paging = ctx.srv->update( culler, loop.camera );
	
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
			//			vm::println("render/fetch/merge = {}/{}/{}", ns0, ns1, ns2 );
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
					if ( local.val.x == 0 && local.val.y == 0 && local.val.z == 0 ) {
						local.val = recv.val;
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

	return render_t;
}

REGISTER_RENDERER( PagingRenderer, "Paging" );
