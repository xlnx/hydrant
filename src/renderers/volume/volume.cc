#include <set>
#include <algorithm>
#include <VMUtils/timer.hpp>
#include <varch/thumbnail.hpp>
#include <hydrant/double_buffering.hpp>
#include <hydrant/octree_culler.hpp>
#include <hydrant/unarchive_pipeline.hpp>
#include "volume.hpp"

using namespace std;
using namespace vol;

#define MAX_SAMPLER_COUNT ( 256 )
#define MAX_BLOCK_COUNT ( 16 )

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	MtArchive const *VolumeRenderer::sample_level( std::size_t level ) const
	{
		auto lp = dataset->meta.sample_levels.find( level );
		if ( lp == dataset->meta.sample_levels.end() ) { return nullptr; }
		// TODO: select archive
		return &lp->second.archives[ 0 ];
	}

	bool VolumeRenderer::init( std::shared_ptr<Dataset> const &dataset,
							   RendererConfig const &cfg )
	{
		if ( !Super::init( dataset, cfg ) ) { return false; }

		auto params = cfg.params.get<VolumeRendererConfig>();
		// shader.render_mode = params.mode == "volume" ? BrmVolume : BrmSolid;
		shader.density = params.density;
		transfer_fn = TransferFn( params.transfer_fn, device );
		shader.transfer_fn = transfer_fn.sampler();

		auto &lvl0_arch = dataset->meta.sample_levels[ 0 ].archives[ 0 ];
		uu.reset( new Unarchiver( dataset->root.resolve( lvl0_arch.path ).resolved() ) );

		auto chebyshev_thumb = std::make_shared<vol::Thumbnail<int>>(
		  dataset->root.resolve( lvl0_arch.thumbnails[ "chebyshev" ] ).resolved() );
		chebyshev = create_texture( chebyshev_thumb );
		shader.chebyshev = chebyshev.sampler();

		vaddr_buf = HostBuffer3D<int>( dim );
		vaddr = Texture3D<int>(
		  Texture3DOptions{}
			.set_device( device )
			.set_dim( dim )
			.set_opts( cufx::Texture::Options::as_array()
						 .set_address_mode( cufx::Texture::AddressMode::Clamp ) ) );

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

	std::shared_ptr<IBuffer3D<unsigned char>>
	  VolumeRenderer::alloc_block_buf( std::size_t pad_bs )
	{
		std::shared_ptr<IBuffer3D<unsigned char>> buf;
		if ( device.has_value() ) {
			buf.reset( new GlobalBuffer3D<unsigned char>( uvec3( pad_bs ), device.value() ) );
		} else {
			buf.reset( new HostBuffer3D<unsigned char>( uvec3( pad_bs ) ) );
		}
		return buf;
	}

	Texture3DOptions VolumeRenderer::block_tex_opts( std::size_t pad_bs ) const
	{
		return Texture3DOptions{}
		  .set_dim( pad_bs )
		  .set_device( device )
		  .set_opts( cufx::Texture::Options{}
					   .set_address_mode( cufx::Texture::AddressMode::Wrap )
					   .set_filter_mode( cufx::Texture::FilterMode::Linear )
					   .set_read_mode( cufx::Texture::ReadMode::NormalizedFloat )
					   .set_normalize_coords( true ) );
	}

	std::vector<VolumeRenderer::LowestLevelBlock>
	  VolumeRenderer::unarchive_lowest_level()
	{
		std::vector<LowestLevelBlock> blocks;

		auto level = dataset->meta.sample_levels.size() - 1;
		auto &arch = *sample_level( level );

		auto nblk_x = ( 1 << level ) * arch.block_size;
		auto nblk_y = dataset->meta.sample_levels[ 0 ].archives[ 0 ].block_size;
		if ( nblk_x % nblk_y != 0 ) {
			throw std::logic_error( "nblk_x % nblk_y != 0" );
		}

		auto nblk_scale = nblk_x / nblk_y;
		auto pad_bs = arch.padding * 2 + arch.block_size;
		auto buf = alloc_block_buf( pad_bs );
		auto opts = block_tex_opts( pad_bs );

		auto k = float( uu->block_size() ) / uu->padded_block_size() / nblk_scale;
		auto b0 = vec3( float( uu->padding() ) / uu->padded_block_size() );

		vector<Idx> idxs;
		idxs.reserve( arch.dim.total() );
		for ( auto idx = Idx{}; idx.z != arch.dim.z; ++idx.z ) {
			for ( idx.y = 0; idx.y != arch.dim.y; ++idx.y ) {
				for ( idx.x = 0; idx.x != arch.dim.x; ++idx.x ) {
					idxs.emplace_back( idx );
				}
			}
		}

		Unarchiver unarchiver( dataset->root.resolve( arch.path ).resolved() );
		size_t nbytes = 0, block_bytes = buf->bytes();
		blocks.reserve( arch.dim.total() );
		unarchiver.unarchive(
		  idxs,
		  [&]( Idx const &idx, VoxelStreamPacket const &pkt ) {
			  pkt.append_to( buf->view_1d() );
			  nbytes += pkt.length;
			  if ( nbytes >= block_bytes ) {
				  Texture3D<unsigned char> block( opts );
				  auto fut = block.source( buf->view_3d() );
				  fut.wait();
				  nbytes = 0;
				  auto base = uvec3( idx.x, idx.y, idx.z ) * nblk_scale;
				  for ( auto dx = uvec3( 0, 0, 0 ); dx.z != nblk_scale; ++dx.z ) {
					  for ( dx.y = 0; dx.y != nblk_scale; ++dx.y ) {
						  for ( dx.x = 0; dx.x != nblk_scale; ++dx.x ) {
							  auto x = dx + base;
							  if ( all( lessThan( x, dim ) ) ) {
								  blocks.emplace_back(
									LowestLevelBlock{}
									  .set_idx( Idx{}
												  .set_x( x.x )
												  .set_y( x.y )
												  .set_z( x.z ) )
									  .set_storage( block )
									  .set_sampler( BlockSampler{}
													  .set_sampler( block.sampler() )
													  .set_mapping(
														BlockSamplerMapping{}
														  .set_k( k )
														  .set_b( b0 + vec3( dx ) * k ) ) ) );
							  }
						  }
					  }
				  }
			  }
		  } );

		return blocks;
	}

	void VolumeRenderer::render_loop( IRenderLoop & loop )
	{
		// TODO: select lowest archive
		auto lowest_blocks = unarchive_lowest_level();
		auto lowest_blkcnt = lowest_blocks.size();
		auto rest_blkcnt = MAX_SAMPLER_COUNT - lowest_blkcnt;

		vector<BlockSampler> host_registry( MAX_SAMPLER_COUNT );
		cufx::MemoryView1D<BlockSampler> host_reg_view( host_registry.data() + lowest_blkcnt,
														rest_blkcnt );

		map<Idx, int> lowest_block_sampler_id;
		for ( int i = 0; i < lowest_blkcnt; ++i ) {
			host_registry[ i ] = lowest_blocks[ i ].sampler;
			lowest_block_sampler_id[ lowest_blocks[ i ].idx ] = i;
		}
		shader.block_sampler = host_registry.data();

		std::shared_ptr<cufx::GlobalMemory> device_registry;
		cufx::MemoryView1D<BlockSampler> device_reg_view;

		if ( device.has_value() ) {
			device_registry.reset( new cufx::GlobalMemory( MAX_SAMPLER_COUNT * sizeof( BlockSampler ),
														   device.value() ) );
			device_reg_view = device_registry->view_1d<BlockSampler>( MAX_SAMPLER_COUNT )
				.slice( lowest_blkcnt, rest_blkcnt );
			cufx::memory_transfer( device_registry->view_1d<BlockSampler>( lowest_blkcnt ),
								   cufx::MemoryView1D<BlockSampler>( host_registry.data(), lowest_blkcnt ) )
			  .launch();
			shader.block_sampler = reinterpret_cast<BlockSampler const *>( device_registry->get() );
		}

		auto update_device_registry_if = [&]() {
			if ( device.has_value() ) {
				cufx::memory_transfer( device_reg_view, host_reg_view )
				  .launch();
			}
		};

		auto basic_vaddr_buf = HostBuffer3D<int>( dim );
		for ( auto &block : lowest_block_sampler_id ) {
			basic_vaddr_buf[ uvec3( block.first.x,
									block.first.y,
									block.first.z ) ] = block.second;
		}
		memcpy( vaddr_buf.data(), basic_vaddr_buf.data(), vaddr_buf.bytes() );

		// cache.reserve( MAX_SAMPLER_COUNT );

		auto film = create_film();

		OctreeCuller culler( exhibit, dim );
		vector<Idx> missing_idxs( MAX_BLOCK_COUNT );
		vector<Idx> redundant_idxs( MAX_BLOCK_COUNT );
		set<Idx> present_idxs;
		std::mutex idxs_mut;

		vector<Texture3D<unsigned char>> block_storage;

		auto &lvl0_arch = dataset->meta.sample_levels[ 0 ].archives[ 0 ];
		auto k = float( lvl0_arch.block_size ) / ( lvl0_arch.block_size + 2 * lvl0_arch.padding );
		auto b = vec3( float( lvl0_arch.padding ) / lvl0_arch.block_size * k );
		auto mapping = BlockSamplerMapping{}.set_k( k ).set_b( b );

		Unarchiver unarchiver( dataset->root.resolve( lvl0_arch.path ).resolved() );
		FnUnarchivePipeline pipeline(
		  unarchiver,
		  [&]( auto &idx, auto &buffer ) {
			  std::unique_lock<std::mutex> lk( idxs_mut );
			  int vaddr_id = -1;
			  if ( block_storage.size() < MAX_BLOCK_COUNT ) {
				  auto opts = Texture3DOptions{}
								.set_dim( unarchiver.padded_block_size() )
								.set_device( device )
								.set_opts( cufx::Texture::Options{}
											 .set_address_mode( cufx::Texture::AddressMode::Wrap )
											 .set_filter_mode( cufx::Texture::FilterMode::Linear )
											 .set_read_mode( cufx::Texture::ReadMode::NormalizedFloat )
											 .set_normalize_coords( true ) );
				  /* skip those lowest blocks */
				  auto storage_id = block_storage.size();
				  vaddr_id = lowest_blkcnt + storage_id;
				  vm::println( "allocate {}", storage_id );
				  block_storage.emplace_back( opts );
			  } else if ( redundant_idxs.size() ) {
				  auto swap_idx = redundant_idxs.back();
				  redundant_idxs.pop_back();
				  present_idxs.erase( swap_idx );
				  auto uvec3_idx = uvec3( swap_idx.x, swap_idx.y, swap_idx.z );
				  auto &swap_vaddr = vaddr_buf[ uvec3_idx ];
				  vaddr_id = swap_vaddr;
				  /* reset that block to lowest sample level */
				  swap_vaddr = basic_vaddr_buf[ uvec3_idx ];
			  } else {
				  vm::println("block {} abandoned", idx);
				  return ;
			  }

			  auto storage_id = vaddr_id - lowest_blkcnt;
			  vm::println( "at {}", storage_id );
			  auto &storage = block_storage[ storage_id ];
			  auto fut = storage.source( buffer.view_3d() );
			  fut.wait();
			  /* TODO: check whether this sampler should be updated */
			  host_reg_view.at( storage_id ) = BlockSampler{}
												 .set_sampler( storage.sampler() )
												 .set_mapping( mapping );
			  //			  update_device_registry_if( storage_id );

			  vaddr_buf[ uvec3( idx.x, idx.y, idx.z ) ] = vaddr_id;
			  present_idxs.insert( idx );
			  vm::println( "u+ {}", idx );
		  },
		  UnarchivePipelineOptions{}
			.set_device( device ) );

		FnDoubleBuffering loop_drv(
		  ImageOptions{}
			.set_device( device )
			.set_resolution( resolution ),
		  loop,
		  [&]( auto &frame, auto frame_idx ) {
			  std::size_t ns = 0, ns1 = 0;

			  //   vm::Timer::Scoped timer( [&]( auto dt ) {
			  // 	  vm::println( "time: {} / {} / {}", dt.ms(),
			  // 				   ns / 1000 / 1000,
			  // 				   ns1 / 1000 / 1000 );
			  //   } );

			  auto require_idxs = culler.cull( loop.camera, MAX_BLOCK_COUNT );
			  {
				  std::unique_lock<std::mutex> lk( idxs_mut );

				  missing_idxs.resize( MAX_BLOCK_COUNT );
				  auto missing_idxs_end = set_difference( require_idxs.begin(), require_idxs.end(),
														  present_idxs.begin(), present_idxs.end(),
														  missing_idxs.begin() );
				  missing_idxs.resize( missing_idxs_end - missing_idxs.begin() );

				  redundant_idxs.resize( MAX_BLOCK_COUNT );
				  auto redundant_idxs_end = set_difference( present_idxs.begin(), present_idxs.end(),
															require_idxs.begin(), require_idxs.end(),
															redundant_idxs.begin() );
				  redundant_idxs.resize( redundant_idxs_end - redundant_idxs.begin() );

				  if ( missing_idxs.size() ) {
					  vm::println( "{}", missing_idxs );
				  }
				  //   for ( auto it = missing_idxs_end; it != redundant_idxs_end; ++it ) {
				  // 	  present_idxs.erase( *it );
				  //   }
				  //   for ( auto &idx : missing_idxs ) {
				  // 	  present_idxs.insert( idx );
				  //   }

				  if (missing_idxs.size()) {
					  pipeline.lock().require( missing_idxs.begin(),
											   missing_idxs.end(),
											   []( auto &idx ) { return 1.f; } );
				  }
			  }

			  update_device_registry_if();

			  vaddr.source( vaddr_buf.data(), false );
			  shader.vaddr = vaddr.sampler();

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

		pipeline.start();
		loop_drv.run();
		pipeline.stop();
	}

	REGISTER_RENDERER( VolumeRenderer, "Volume" );
}

VM_END_MODULE()
