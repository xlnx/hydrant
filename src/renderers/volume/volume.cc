#include <set>
#include <algorithm>
#include <VMUtils/timer.hpp>
#include <varch/thumbnail.hpp>
#include <hydrant/double_buffering.hpp>
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

		auto params = cfg.params.get<VolumeRendererConfig>();
		// shader.render_mode = params.mode == "volume" ? BrmVolume : BrmSolid;
		shader.density = params.density;
		shader.mode = params.mode;
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

	cufx::Image<> VolumeRenderer::offline_render( Camera const &camera )
	{
		auto film = create_film();

		// auto k = float( uu->block_size() ) / uu->padded_block_size();
		// auto b = vec3( float( uu->padding() ) / uu->block_size() * k );
		// auto mapping = BlockSamplerMapping{}
		// 				 .set_k( k )
		// 				 .set_b( b );

		// auto pad_bs = uu->padded_block_size();
		// auto block_bytes = pad_bs * pad_bs * pad_bs;

		// std::shared_ptr<IBuffer3D<unsigned char>> buf;
		// if ( device.has_value() ) {
		// 	buf.reset( new GlobalBuffer3D<unsigned char>( uvec3( pad_bs ), device.value() ) );
		// } else {
		// 	buf.reset( new HostBuffer3D<unsigned char>( uvec3( pad_bs ) ) );
		// }

		// auto et = exhibit.get_matrix();

		// std::size_t ns = 0, ns1 = 0;
		// {
		// 	vm::Timer::Scoped timer( [&]( auto dt ) {
		// 		vm::println( "time: {} / {} / {}", dt.ms(),
		// 					 ns / 1000 / 1000,
		// 					 ns1 / 1000 / 1000 );
		// 	} );

		// 	glm::vec3 cp = et * glm::vec4( camera.position, 1 );

		// 	std::sort( pidx.begin(), pidx.end(),
		// 			   [&]( int x, int y ) {
		// 				   return glm::distance( block_ccs[ x ], cp ) <
		// 						  glm::distance( block_ccs[ y ], cp );
		// 			   } );

		// 	for ( int i = 0; i < pidx.size(); i += MAX_SAMPLER_COUNT ) {
		// 		vector<Idx> idxs;
		// 		for ( int j = i; j < i + MAX_SAMPLER_COUNT && j < pidx.size(); ++j ) {
		// 			idxs.emplace_back( block_idxs[ pidx[ j ] ] );
		// 		}

		// 		int nbytes = 0, blkid = 0;
		// 		memset( vaddr_buf.data(), -1, vaddr_buf.bytes() );

		// 		{
		// 			vm::Timer::Scoped timer( [&]( auto dt ) {
		// 				ns += dt.ns().cnt();
		// 			} );

		// 			uu->unarchive(
		// 			  idxs,
		// 			  [&]( Idx const &idx, VoxelStreamPacket const &pkt ) {
		// 				  pkt.append_to( buf->view_1d() );
		// 				  nbytes += pkt.length;
		// 				  if ( nbytes >= block_bytes ) {
		// 					  auto fut = cache[ blkid ].source( buf->view_3d() );
		// 					  fut.wait();
		// 					  vaddr_buf[ glm::vec3( idx.x, idx.y, idx.z ) ] = blkid;
		// 					  nbytes = 0;
		// 					  blkid += 1;
		// 					  //   }
		// 				  }
		// 			  } );
		// 		}

		// 		vaddr.source( vaddr_buf.data(), false );
		// 		shader.vaddr = vaddr.sampler();

		// 		// vm::println( "{}", cache_texs.size() );
		// 		for ( int j = 0; j != cache.size(); ++j ) {
		// 			shader.block_sampler[ j ] = BlockSampler{}
		// 										  .set_sampler( cache[ j ].sampler() )
		// 										  .set_mapping( mapping );
		// 		}

		// 		{
		// 			vm::Timer::Scoped timer( [&]( auto dt ) {
		// 				ns1 += dt.ns().cnt();
		// 			} );

		// 			if ( i == 0 ) {
		// 				raycaster.cast( exhibit,
		// 								camera,
		// 								film.view(),
		// 								shader,
		// 								RaycastingOptions{}
		// 								  .set_device( device ) );
		// 			} else {
		// 				raycaster.cast( film.view(),
		// 								shader,
		// 								RaycastingOptions{}
		// 								  .set_device( device ) );
		// 			}
		// 		}
		// 	}
		// }

		return film.fetch_data().dump();
	}

	void VolumeRenderer::render_loop( IRenderLoop & loop )
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

				  raycaster.cast( exhibit,
								  loop.camera,
								  film.view(),
								  shader,
								  opts );

				  raycaster.cast( film.view(),
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
