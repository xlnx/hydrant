#include <set>
#include <algorithm>
#include <glog/logging.h>
#include <hydrant/bridge/texture_3d.hpp>
#include <hydrant/bridge/buffer_3d.hpp>
#include <hydrant/unarchiver.hpp>
#include <hydrant/paging/unarchive_pipeline.hpp>
#include <hydrant/paging/rt_block_paging.hpp>

#define MAX_SAMPLER_COUNT ( 32768 )

VM_BEGIN_MODULE( hydrant )

using namespace std;
using namespace vol;

struct LowestLevelBlock
{
	VM_DEFINE_ATTRIBUTE( Idx, idx );
	VM_DEFINE_ATTRIBUTE( Texture3D<unsigned char>, storage );
	VM_DEFINE_ATTRIBUTE( BlockSampler, sampler );
};

struct RtBlockPagingRegistry
{
	vector<BlockSampler> host_registry;
	cufx::MemoryView1D<BlockSampler> host_reg_view;

	shared_ptr<cufx::GlobalMemory> device_registry;
	cufx::MemoryView1D<BlockSampler> device_reg_view;

	map<Idx, int> lowest_block_sampler_id;

public:
	RtBlockPagingRegistry( BlockPaging &client,
						   vector<LowestLevelBlock> const &lowest_blocks,
						   vm::Option<cufx::Device> const &device )
	{
		auto lowest_blkcnt = lowest_blocks.size();
		auto rest_blkcnt = MAX_SAMPLER_COUNT - lowest_blkcnt;
		client.lowest_blkcnt = lowest_blkcnt;

		host_registry.resize( MAX_SAMPLER_COUNT );
		host_reg_view = cufx::MemoryView1D<BlockSampler>( host_registry.data() + lowest_blkcnt,
														  rest_blkcnt );
		for ( int i = 0; i < lowest_blkcnt; ++i ) {
			host_registry[ i ] = lowest_blocks[ i ].sampler;
			lowest_block_sampler_id[ lowest_blocks[ i ].idx ] = i;
		}
		client.block_sampler = host_registry.data();

		if ( device.has_value() ) {
			device_registry.reset( new cufx::GlobalMemory( MAX_SAMPLER_COUNT * sizeof( BlockSampler ),
														   device.value() ) );
			device_reg_view = device_registry->view_1d<BlockSampler>( MAX_SAMPLER_COUNT )
								.slice( lowest_blkcnt, rest_blkcnt );
			cufx::memory_transfer( device_registry->view_1d<BlockSampler>( lowest_blkcnt ),
								   cufx::MemoryView1D<BlockSampler>( host_registry.data(), lowest_blkcnt ) )
			  .launch();
			client.block_sampler = reinterpret_cast<BlockSampler const *>( device_registry->get() );
		}
	}

	void update()
	{
		if ( device_registry ) {
			cufx::memory_transfer( device_reg_view, host_reg_view )
			  .launch();
		}
	}
};

struct RtBlockPagingServerImpl
{
	RtBlockPagingServerImpl( RtBlockPagingServerOptions const &opts );

public:
	shared_ptr<IBuffer3D<unsigned char>> alloc_block_buf( size_t pad_bs );

	void update( OctreeCuller &culler, Camera const &camera );

	void unarchive_lowest_level();

public:
	RtBlockPagingServerOptions opts;
	size_t max_block_count;

	vector<LowestLevelBlock> lowest_blocks;

	Texture3D<int> vaddr;
	HostBuffer3D<int> vaddr_buf;
	HostBuffer3D<int> basic_vaddr_buf;

	unique_ptr<RtBlockPagingRegistry> registry;
	unique_ptr<Unarchiver> unarchiver;
	unique_ptr<FnUnarchivePipeline> pipeline;

	vector<Idx> missing_idxs;
	vector<Idx> redundant_idxs;
	set<Idx> present_idxs;
	mutex idxs_mut;

	vector<Texture3D<unsigned char>> block_storage;

	BlockPaging client;
};

shared_ptr<IBuffer3D<unsigned char>> RtBlockPagingServerImpl::alloc_block_buf( std::size_t pad_bs )
{
	shared_ptr<IBuffer3D<unsigned char>> buf;
	if ( opts.device.has_value() ) {
		buf.reset( new GlobalBuffer3D<unsigned char>( uvec3( pad_bs ), opts.device.value() ) );
	} else {
		buf.reset( new HostBuffer3D<unsigned char>( uvec3( pad_bs ) ) );
	}
	return buf;
}

void RtBlockPagingServerImpl::unarchive_lowest_level()
{
	auto &lvl0 = opts.dataset->meta.sample_levels[ 0 ];
	
	auto level = opts.dataset->meta.sample_levels.size() - 1;
	auto &arch = opts.dataset->meta.sample_levels[ level ];

	auto nblk_x = ( 1 << level ) * opts.dataset->meta.block_size;
	auto nblk_y = opts.dataset->meta.block_size;
	if ( nblk_x % nblk_y != 0 ) {
		LOG( FATAL ) << "nblk_x % nblk_y != 0";
	}

	auto nblk_scale = nblk_x / nblk_y;
	auto pad_bs = opts.dataset->meta.padding * 2 + opts.dataset->meta.block_size;
	auto buf = alloc_block_buf( pad_bs );
	auto storage_opts = Texture3DOptions{}
						  .set_dim( pad_bs )
						  .set_device( opts.device )
						  .set_opts( opts.storage_opts );

	auto bs_0 = opts.dataset->meta.block_size;
	auto pad_0 = opts.dataset->meta.padding;
	auto pad_bs_0 = opts.dataset->meta.block_size + 2 * pad_0;
	auto k = float( bs_0 ) / pad_bs_0 / nblk_scale;
	auto b0 = vec3( float( pad_0 ) / pad_bs_0 );

	vector<Idx> idxs;
	idxs.reserve( arch.dim.total() );
	for ( auto idx = Idx{}; idx.z != arch.dim.z; ++idx.z ) {
		for ( idx.y = 0; idx.y != arch.dim.y; ++idx.y ) {
			for ( idx.x = 0; idx.x != arch.dim.x; ++idx.x ) {
				idxs.emplace_back( idx );
			}
		}
	}

	Unarchiver unarchiver( UnarchiverOptions{}
                                    .set_path( opts.dataset->root.resolve( arch.path ).resolved() )
                                    .set_device( opts.device ) );
	size_t nbytes = 0, block_bytes = buf->bytes();
	lowest_blocks.reserve( arch.dim.total() );
	unarchiver.unarchive(
	  idxs,
	  [&]( Idx const &idx, VoxelStreamPacket const &pkt ) {
		  pkt.append_to( buf->view_1d() );
		  nbytes += pkt.length;
		  if ( nbytes >= block_bytes ) {
			  Texture3D<unsigned char> block( storage_opts );
			  auto fut = block.source( buf->view_3d() );
			  fut.wait();
			  nbytes = 0;
			  auto base = uvec3( idx.x, idx.y, idx.z ) * nblk_scale;
			  for ( auto dx = uvec3( 0, 0, 0 ); dx.z != nblk_scale; ++dx.z ) {
				  for ( dx.y = 0; dx.y != nblk_scale; ++dx.y ) {
					  for ( dx.x = 0; dx.x != nblk_scale; ++dx.x ) {
						  auto x = dx + base;
						  if ( all( lessThan( x, opts.dim ) ) ) {
							  lowest_blocks.emplace_back(
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
}

RtBlockPagingServerImpl::RtBlockPagingServerImpl( RtBlockPagingServerOptions const &opts ) :
  opts( opts )
{
	auto &lvl0 = opts.dataset->meta.sample_levels[ 0 ];

	basic_vaddr_buf = HostBuffer3D<int>( opts.dim );
	vaddr_buf = HostBuffer3D<int>( opts.dim );
	vaddr = Texture3D<int>(
	  Texture3DOptions{}
		.set_device( opts.device )
		.set_dim( opts.dim )
		.set_opts( cufx::Texture::Options::as_array()
					 .set_address_mode( cufx::Texture::AddressMode::Clamp ) ) );

	unarchive_lowest_level();
	registry.reset( new RtBlockPagingRegistry( client, lowest_blocks, opts.device ) );

	for ( auto &block : registry->lowest_block_sampler_id ) {
		basic_vaddr_buf[ uvec3( block.first.x,
								block.first.y,
								block.first.z ) ] = block.second;
	}
	memcpy( vaddr_buf.data(), basic_vaddr_buf.data(), vaddr_buf.bytes() );

	auto bs = opts.dataset->meta.block_size;
	auto pad = opts.dataset->meta.padding;
	auto pad_bs = opts.dataset->meta.block_size + 2 * pad;
	auto k = float( bs ) / pad_bs;
	auto b = vec3( float( pad ) / bs * k );
	auto mapping = BlockSamplerMapping{}.set_k( k ).set_b( b );
	auto storage_opts = Texture3DOptions{}
						  .set_dim( pad_bs )
						  .set_device( opts.device )
						  .set_opts( opts.storage_opts );

	auto mem_limit_bytes = uint64_t(opts.mem_limit_mb) * 1024 * 1024;
	auto block_bytes = pad_bs * pad_bs * pad_bs;
	max_block_count = mem_limit_bytes / block_bytes;
	max_block_count = std::min( max_block_count, MAX_SAMPLER_COUNT - lowest_blocks.size() );

	LOG( INFO ) << vm::fmt( "MEM_LIMIT_MB = {}", opts.mem_limit_mb );
	LOG( INFO ) << vm::fmt( "MEM_LIMIT_BYTES = {}", mem_limit_bytes );
	LOG( INFO ) << vm::fmt( "BLOCK_BYTES = {}", block_bytes );
	LOG( INFO ) << vm::fmt( "MAX_BLOCK_COUNT = {}", max_block_count );

	unarchiver.reset( new Unarchiver( UnarchiverOptions{}
                                    .set_path( opts.dataset->root.resolve( lvl0.path ).resolved() )
                                    .set_device( opts.device ) ) );
	pipeline.reset(
	  new FnUnarchivePipeline(
		*unarchiver,
		[&, storage_opts, mapping]( auto &idx, auto &buffer ) {
			unique_lock<mutex> lk( idxs_mut );
			int vaddr_id = -1;
			if ( present_idxs.count( idx ) ) {
				LOG( WARNING ) << vm::fmt( "abandoned {}", idx );
				return;
			}
			if ( block_storage.size() < max_block_count ) {
				/* skip those lowest blocks */
				auto storage_id = block_storage.size();
				vaddr_id = lowest_blocks.size() + storage_id;
				// vm::println( "allocate {}", storage_id );
				block_storage.emplace_back( storage_opts );
			} else if ( redundant_idxs.size() ) {
				auto swap_idx = redundant_idxs.back();
				redundant_idxs.pop_back();
				//				LOG( INFO ) << vm::fmt( "swap +{} -{}", idx, swap_idx );
				present_idxs.erase( swap_idx );
				auto uvec3_idx = uvec3( swap_idx.x, swap_idx.y, swap_idx.z );
				auto &swap_vaddr = vaddr_buf[ uvec3_idx ];
				vaddr_id = swap_vaddr;
				/* reset that block to lowest sample level */
				swap_vaddr = basic_vaddr_buf[ uvec3_idx ];
			} else {
				LOG( WARNING ) << vm::fmt( "artifact {}", idx );
				return;
			}

			auto storage_id = vaddr_id - lowest_blocks.size();
			// vm::println( "at {}", storage_id );
			auto &storage = block_storage[ storage_id ];
			auto fut = storage.source( buffer.view_3d() );
			fut.wait();
			/* TODO: check whether this sampler should be updated */
			registry->host_reg_view.at( storage_id ) = BlockSampler{}
														 .set_sampler( storage.sampler() )
														 .set_mapping( mapping );
			//			  update_device_registry_if( storage_id );

			vaddr_buf[ uvec3( idx.x, idx.y, idx.z ) ] = vaddr_id;
			present_idxs.insert( idx );
			// vm::println( "u+ {}", idx );
		},
		UnarchivePipelineOptions{}
		  .set_device( opts.device ) ) );
}

void RtBlockPagingServerImpl::update( OctreeCuller &culler, Camera const &camera )
{
	std::function<float( const vol::Idx & )> dist_fn;
	auto &require_idxs = culler.cull( camera, &dist_fn, max_block_count );
	{
		std::unique_lock<std::mutex> lk( idxs_mut );

		missing_idxs.resize( max_block_count );
		auto missing_idxs_end = set_difference( require_idxs.begin(), require_idxs.end(),
												present_idxs.begin(), present_idxs.end(),
												missing_idxs.begin() );
		missing_idxs.resize( missing_idxs_end - missing_idxs.begin() );

		redundant_idxs.resize( max_block_count );
		auto redundant_idxs_end = set_difference( present_idxs.begin(), present_idxs.end(),
												  require_idxs.begin(), require_idxs.end(),
												  redundant_idxs.begin() );
		redundant_idxs.resize( redundant_idxs_end - redundant_idxs.begin() );

		if ( missing_idxs.size() ) {
			std::sort( missing_idxs.begin(), missing_idxs.end(),
					   [&]( auto &a,  auto &b ){ return dist_fn( a ) < dist_fn( b ); } );
			pipeline->lock().require( missing_idxs );
		}
	}

	registry->update();

	vaddr.source( vaddr_buf.data(), false );
	client.vaddr = vaddr.sampler();
}

VM_EXPORT
{
	RtBlockPagingServer::RtBlockPagingServer( RtBlockPagingServerOptions const &opts ) :
	  _( new RtBlockPagingServerImpl( opts ) )
	{
	}

	RtBlockPagingServer::~RtBlockPagingServer()
	{
	}

	BlockPaging RtBlockPagingServer::update( OctreeCuller & culler, Camera const &camera )
	{
		_->update( culler, camera );
		return _->client;
	}

	void RtBlockPagingServer::start()
	{
		_->pipeline->start();
	}

	void RtBlockPagingServer::stop()
	{
		_->pipeline->stop();
	}
}

VM_END_MODULE()
