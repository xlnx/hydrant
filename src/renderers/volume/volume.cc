#include <VMUtils/timer.hpp>
#include <varch/thumbnail.hpp>
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

		auto my_cfg = cfg.params.get<VolumeRendererConfig>();
		// shader.render_mode = my_cfg.mode == "volume" ? BrmVolume : BrmSolid;
		// shader.density = my_cfg.density;

		auto &lvl0_arch = dataset->meta.sample_levels[ 0 ].archives[ 0 ];
		uu.reset( new Unarchiver( dataset->root.resolve( lvl0_arch.path ).resolved() ) );

		shader.cache_du.x = float( uu->padding() ) / uu->block_size();
		shader.cache_du.y = float( uu->block_size() ) / uu->padded_block_size();

		auto chebyshev_thumb = std::make_shared<vol::Thumbnail<int>>(
		  dataset->root.resolve( lvl0_arch.thumbnails[ "chebyshev" ] ).resolved() );
		chebyshev = create_texture( chebyshev_thumb );
		shader.chebyshev = chebyshev.sampler();

		present_buf = HostBuffer3D<int>( dim );
		present = Texture3D<int>(
		  Texture3DOptions{}
			.set_device( device )
			.set_dim( dim )
			.set_opts( cufx::Texture::Options::as_array()
						 .set_address_mode( cufx::Texture::AddressMode::Clamp ) ) );

		/* absent buffer */
#pragma region

		// auto wg_cnt = 32 * 32;
		// shader.wg_max_emit_cnt = 8;
		// shader.wg_len_bytes = sizeof( int ) +
		// 					  shader.wg_max_emit_cnt * sizeof( glm::uvec3 );
		// auto absent_glob = device.alloc_global( shader.wg_len_bytes * wg_cnt );
		// vector<char> absent( absent_glob.size() );
		// shader.absent_buf = absent_glob.view_1d<char>( absent_glob.size() );

#pragma endregion

		chebyshev_thumb->iterate_3d(
		  [&]( vol::Idx const &idx ) {
			  if ( !( *chebyshev_thumb )[ idx ] ) {
				  block_idxs.emplace_back( idx );
			  }
		  } );
		vm::println( "{}", block_idxs );
		block_ccs.resize( block_idxs.size() );
		std::transform( block_idxs.begin(), block_idxs.end(), block_ccs.begin(),
						[]( Idx const &idx ) { return glm::vec3( idx.x, idx.y, idx.z ) + 0.5f; } );
		pidx.resize( block_idxs.size() );
		for ( int i = 0; i != pidx.size(); ++i ) { pidx[ i ] = i; }
		vm::println( "{}", block_idxs.size() );

		cache.reserve( MAX_CACHE_SIZE );
		for ( int i = 0; i != MAX_CACHE_SIZE; ++i ) {
			cache.emplace_back(
			  Texture3DOptions{}
				.set_dim( uu->padded_block_size() )
				.set_device( device )
				.set_opts( cufx::Texture::Options{}
							 .set_address_mode( cufx::Texture::AddressMode::Wrap )
							 .set_filter_mode( cufx::Texture::FilterMode::Linear )
							 .set_read_mode( cufx::Texture::ReadMode::NormalizedFloat )
							 .set_normalize_coords( true ) ) );
		}

		return true;
	}

	void VolumeRenderer::offline_render( std::string const &dst_path,
										 Camera const &camera )
	{
		// cufx::MemoryView3D<int> chebyshev_view( chebyshev.data(), thumbnail_view_info, thumbnail_extent );
		// cufx::memory_transfer( sampler_arr, block_view_3d, cudaPos{ 0, 0, 0 } ).launch();
		// cufx::Texture sampler_texture( sampler_arr, cufx::Texture::Options::as_array() );

		auto pad_bs = uu->padded_block_size();
		auto block_bytes = pad_bs * pad_bs * pad_bs;

		std::shared_ptr<Buffer3D<unsigned char>> buf;
		if ( device.has_value() ) {
			buf.reset( new GlobalBuffer3D<unsigned char>( uvec3( pad_bs ), device.value() ) );
		} else {
			buf.reset( new HostBuffer3D<unsigned char>( uvec3( pad_bs ) ) );
		}

		auto et = exhibit.get_matrix();

		std::size_t ns = 0, ns1 = 0;
		{
			vm::Timer::Scoped timer( [&]( auto dt ) {
				vm::println( "time: {} / {} / {}", dt.ms(),
							 ns / 1000 / 1000,
							 ns1 / 1000 / 1000 );
			} );

			glm::vec3 cp = et * glm::vec4( camera.position, 1 );

			std::sort( pidx.begin(), pidx.end(),
					   [&]( int x, int y ) {
						   return glm::distance( block_ccs[ x ], cp ) <
								  glm::distance( block_ccs[ y ], cp );
					   } );

			for ( int i = 0; i < pidx.size(); i += MAX_CACHE_SIZE ) {
				vector<Idx> idxs;
				for ( int j = i; j < i + MAX_CACHE_SIZE && j < pidx.size(); ++j ) {
					idxs.emplace_back( block_idxs[ pidx[ j ] ] );
				}

				int nbytes = 0, blkid = 0;
				memset( present_buf.data(), -1, present_buf.bytes() );

				{
					vm::Timer::Scoped timer( [&]( auto dt ) {
						ns += dt.ns().cnt();
					} );

					uu->unarchive(
					  idxs,
					  [&]( Idx const &idx, VoxelStreamPacket const &pkt ) {
						  pkt.append_to( buf->view_1d() );
						  nbytes += pkt.length;
						  if ( nbytes >= block_bytes ) {
							  auto fut = cache[ blkid ].source( buf->view_3d() );
							  fut.wait();
							  present_buf[ glm::vec3( idx.x, idx.y, idx.z ) ] = blkid;
							  nbytes = 0;
							  blkid += 1;
							  //   }
						  }
					  } );
				}

				present.source( present_buf.data(), false );
				shader.present = present.sampler();

				// vm::println( "{}", cache_texs.size() );
				for ( int j = 0; j != cache.size(); ++j ) {
					shader.cache_tex[ j ] = cache[ j ].sampler();
				}

				{
					vm::Timer::Scoped timer( [&]( auto dt ) {
						ns1 += dt.ns().cnt();
					} );

					if ( i == 0 ) {
						raycaster.cast( exhibit,
										camera,
										film.view(),
										shader,
										RaycastingOptions{}
										  .set_device( device ) );
					} else {
						raycaster.cast( film.view(),
										shader,
										RaycastingOptions{}
										  .set_device( device ) );
					}
				}
			}
		}

		film.fetch_data().dump( dst_path );
	}

	REGISTER_RENDERER( VolumeRenderer, "Volume" );
}

VM_END_MODULE()
