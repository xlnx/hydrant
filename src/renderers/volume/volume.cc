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

		image = CudaImage<typename Shader::Pixel>( cfg.resolution, device );

		auto &lvl0_arch = dataset->meta.sample_levels[ 0 ].archives[ 0 ];
		uu.reset( new Unarchiver( dataset->root.resolve( lvl0_arch.path ).resolved() ) );

		glm::uvec3 dim = { uu->dim().x, uu->dim().y, uu->dim().z };
		glm::uvec3 bdim = { uu->padded_block_size(), uu->padded_block_size(), uu->padded_block_size() };

		glm::vec3 raw = { uu->raw().x, uu->raw().y, uu->raw().z };
		glm::vec3 f_dim = raw / float( uu->block_size() );
		// glm::vec3 max = { dim.x, dim.y, dim.z };
		exhibit = Exhibit{}
					.set_center( f_dim / 2.f )
					.set_size( f_dim );

		shader.bbox = Box3D{ { 0, 0, 0 }, f_dim };
		shader.step = 1e-2f * f_dim.x / 4.f;

		shader.cache_du.x = float( uu->padding() ) / uu->block_size();
		shader.cache_du.y = float( uu->block_size() ) / uu->padded_block_size();

		Thumbnail chebyshev_thumb( dataset->root.resolve( lvl0_arch.thumbnails[ "chebyshev" ] ).resolved() );
		chebyshev = ConstTexture3D<float>(
		  dim, chebyshev_thumb.data(),
		  cufx::Texture::Options::as_array()
			.set_address_mode( cufx::Texture::AddressMode::Clamp ),
		  device );
		shader.chebyshev_tex = chebyshev.value().get();

		present_buf = Buffer3D<int>( dim );
		present = ConstTexture3D<int>(
		  dim, present_buf.value().data(),
		  cufx::Texture::Options::as_array()
			.set_address_mode( cufx::Texture::AddressMode::Clamp ),
		  device );

		/* absent buffer */
#pragma region

		auto wg_cnt = 32 * 32;
		shader.wg_max_emit_cnt = 8;
		shader.wg_len_bytes = sizeof( int ) +
							  shader.wg_max_emit_cnt * sizeof( glm::uvec3 );
		auto absent_glob = device.alloc_global( shader.wg_len_bytes * wg_cnt );
		vector<char> absent( absent_glob.size() );
		shader.absent_buf = absent_glob.view_1d<char>( absent_glob.size() );

#pragma endregion

		chebyshev_thumb.iterate_3d(
		  [&]( vol::Idx const &idx ) {
			  if ( !chebyshev_thumb[ idx ] ) {
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

		return true;
	}

	void VolumeRenderer::offline_render( std::string const &dst_path,
										 Camera const &camera )
	{
		/* block buffer */

		auto pad_bs = uu->padded_block_size();
		auto block_bytes = pad_bs * pad_bs * pad_bs;
		auto block_glob = device.alloc_global( block_bytes );
		auto block_view_1d = block_glob.view_1d<unsigned char>( block_bytes );
		auto block_view_info = cufx::MemoryView2DInfo{}
								 .set_stride( pad_bs * sizeof( unsigned char ) )
								 .set_width( pad_bs )
								 .set_height( pad_bs );
		auto block_extent = cufx::Extent{}
							  .set_width( pad_bs )
							  .set_height( pad_bs )
							  .set_depth( pad_bs );
		auto block_view_3d = block_glob.view_3d<unsigned char>( block_view_info, block_extent );
		vector<cufx::Array3D<unsigned char>> cache_block_arr;
		for ( int i = 0; i != MAX_CACHE_SIZE; ++i ) {
			cache_block_arr.emplace_back( device.alloc_arraynd<unsigned char, 3>( block_extent ) );
		}
		// cufx::MemoryView3D<int> chebyshev_view( chebyshev.data(), thumbnail_view_info, thumbnail_extent );
		// cufx::memory_transfer( sampler_arr, block_view_3d, cudaPos{ 0, 0, 0 } ).launch();
		// cufx::Texture sampler_texture( sampler_arr, cufx::Texture::Options::as_array() );

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
				vector<cufx::Texture> cache_texs;
				memset( present_buf.value().data(), -1, present_buf.value().bytes() );

				{
					vm::Timer::Scoped timer( [&]( auto dt ) {
						ns += dt.ns().cnt();
					} );

					uu->unarchive(
					  idxs,
					  [&]( Idx const &idx, VoxelStreamPacket const &pkt ) {
						  pkt.append_to( block_view_1d );
						  nbytes += pkt.length;
						  if ( nbytes >= block_bytes ) {
							  cufx::memory_transfer( cache_block_arr[ blkid ], block_view_3d ).launch();
							  //   if ( blkid == 0 ) {
							  cache_texs.emplace_back( cache_block_arr[ blkid ],
													   cufx::Texture::Options{}
														 .set_address_mode( cufx::Texture::AddressMode::Wrap )
														 .set_filter_mode( cufx::Texture::FilterMode::Linear )
														 .set_read_mode( cufx::Texture::ReadMode::NormalizedFloat )
														 .set_normalize_coords( true ) );
							  present_buf.value()[ glm::vec3( idx.x, idx.y, idx.z ) ] = blkid;
							  nbytes = 0;
							  blkid += 1;
							  //   }
						  }
					  } );
				}

				shader.present_tex = present.value().update();

				// vm::println( "{}", cache_texs.size() );
				for ( int j = 0; j != cache_texs.size(); ++j ) {
					shader.cache_tex[ j ] = cache_texs[ j ];
				}

				{
					vm::Timer::Scoped timer( [&]( auto dt ) {
						ns1 += dt.ns().cnt();
					} );

					if ( i == 0 ) {
						raycaster.cast( exhibit, camera, image.value().view(), shader );
					} else {
						raycaster.cast( image.value().view(), shader );
					}
				}
			}

			image.value().view().copy_from_device().launch();
		}

		image.value().get().dump( dst_path );
	}
}

VM_END_MODULE()
