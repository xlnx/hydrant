#include <hydrant/bridge/texture_3d.hpp>
#include <hydrant/bridge/buffer_3d.hpp>
#include <hydrant/unarchiver.hpp>
#include <hydrant/paging/lossless_block_paging.hpp>

#define MAX_SAMPLER_COUNT ( 4096 )

VM_BEGIN_MODULE( hydrant )

using namespace std;
using namespace vol;

struct LosslessBlockPagingServerImpl
{
	LosslessBlockPagingServerImpl( LosslessBlockPagingServerOptions const &opts )
	{
		auto lvl0_arch = &opts.dataset->meta.sample_levels[ 0 ].archives[ 0 ];
		auto bs = lvl0_arch->block_size;
		auto pad = lvl0_arch->padding;
		auto pad_bs = bs + pad * 2;
		auto k = float( bs ) / pad_bs;
		auto b = vec3( float( pad ) / bs * k );
		auto mem_limit_bytes = opts.mem_limit_mb * 1024 * 1024;
		auto dim = uvec3( lvl0_arch->dim.x, lvl0_arch->dim.y, lvl0_arch->dim.z );
		auto storage_opts = Texture3DOptions{}
							  .set_dim( pad_bs )
							  .set_device( opts.device )
							  .set_opts( opts.storage_opts );

		block_bytes = pad_bs * pad_bs * pad_bs;
		batch_size = mem_limit_bytes / block_bytes;
		batch_size = std::min( batch_size, MAX_SAMPLER_COUNT );
		vm::println( "MEM_LIMIT_BYTES = {}", mem_limit_bytes );
		vm::println( "BLOCK_BYTES = {}", block_bytes );
		vm::println( "MAX_BLOCK_COUNT = {}", batch_size );
		mapping = BlockSamplerMapping{}
					.set_k( k )
					.set_b( b );

		if ( opts.device.has_value() ) {
			buf.reset( new GlobalBuffer3D<unsigned char>( uvec3( pad_bs ), opts.device.value() ) );
		} else {
			buf.reset( new HostBuffer3D<unsigned char>( uvec3( pad_bs ) ) );
		}

		vaddr_buf = HostBuffer3D<int>( dim );
		vaddr = Texture3D<int>(
		  Texture3DOptions{}
			.set_device( opts.device )
			.set_dim( dim )
			.set_opts( cufx::Texture::Options::as_array()
						 .set_address_mode( cufx::Texture::AddressMode::Clamp ) ) );
		uu.reset( new Unarchiver( opts.dataset->root.resolve( lvl0_arch->path ).resolved() ) );

		host_registry.resize( batch_size );
		host_reg_view = cufx::MemoryView1D<BlockSampler>( host_registry.data(), batch_size );
		client.block_sampler = host_registry.data();

		if ( opts.device.has_value() ) {
			device_registry.reset( new cufx::GlobalMemory( batch_size * sizeof( BlockSampler ),
														   opts.device.value() ) );
			device_reg_view = device_registry->view_1d<BlockSampler>( batch_size );
			client.block_sampler = reinterpret_cast<BlockSampler const *>( device_registry->get() );
		}

		for ( auto i = 0; i != batch_size; ++i ) {
			block_storage.emplace_back( storage_opts );
		}
		client.lowest_blkcnt = 0;
	}

	void update()
	{
		if ( device_registry ) {
			cufx::memory_transfer( device_reg_view, host_reg_view )
			  .launch();
		}

		vaddr.source( vaddr_buf.data(), false );
		client.vaddr = vaddr.sampler();
	}

public:
	size_t block_bytes;
	int batch_size;
	BlockSamplerMapping mapping;
	shared_ptr<IBuffer3D<unsigned char>> buf;
	HostBuffer3D<int> vaddr_buf;
	Texture3D<int> vaddr;
	shared_ptr<Unarchiver> uu;

	vector<BlockSampler> host_registry;
	cufx::MemoryView1D<BlockSampler> host_reg_view;

	shared_ptr<cufx::GlobalMemory> device_registry;
	cufx::MemoryView1D<BlockSampler> device_reg_view;

	vector<Texture3D<unsigned char>> block_storage;

	BlockPaging client;
};

VM_EXPORT
{
	LosslessBlockPagingServer::LosslessBlockPagingServer( LosslessBlockPagingServerOptions const &opts ) :
	  _( new LosslessBlockPagingServerImpl( opts ) )
	{
	}

	LosslessBlockPagingServer::~LosslessBlockPagingServer()
	{
	}

	LosslessBlockPagingState LosslessBlockPagingServer::start( OctreeCuller & culler,
															   Camera const &camera,
															   mat4 const &et )
	{
		LosslessBlockPagingState state;
		state.block_idxs = culler.cull( camera );
		vector<vec3> block_ccs( state.block_idxs.size() );
		transform( state.block_idxs.begin(), state.block_idxs.end(), block_ccs.begin(),
				   []( Idx const &idx ) { return vec3( idx.x, idx.y, idx.z ) + 0.5f; } );
		state.pidx.resize( state.block_idxs.size() );
		for ( int i = 0; i != state.pidx.size(); ++i ) { state.pidx[ i ] = i; }
		vec3 cp = et * vec4( camera.position, 1 );
		sort( state.pidx.begin(), state.pidx.end(),
			  [&]( int x, int y ) {
				  return distance( block_ccs[ x ], cp ) <
						 distance( block_ccs[ y ], cp );
			  } );
		state.self = _.get();
		return state;
	}

	bool LosslessBlockPagingState::next( BlockPaging & paging )
	{
		if ( i >= pidx.size() ) return false;

		vector<Idx> idxs;
		for ( int j = i; j < i + self->batch_size && j < pidx.size(); ++j ) {
			idxs.emplace_back( block_idxs[ pidx[ j ] ] );
		}

		int nbytes = 0, blkid = 0;
		memset( self->vaddr_buf.data(), -1, self->vaddr_buf.bytes() );

		self->uu->unarchive(
		  idxs,
		  [&]( Idx const &idx, VoxelStreamPacket const &pkt ) {
			  pkt.append_to( self->buf->view_1d() );
			  nbytes += pkt.length;
			  if ( nbytes >= self->block_bytes ) {
				  auto &storage = self->block_storage[ blkid ];
				  auto fut = storage.source( self->buf->view_3d() );
				  fut.wait();

				  self->host_reg_view.at( blkid ) = BlockSampler{}
													  .set_sampler( storage.sampler() )
													  .set_mapping( self->mapping );

				  self->vaddr_buf[ glm::vec3( idx.x, idx.y, idx.z ) ] = blkid;
				  nbytes = 0;
				  blkid += 1;
			  }
		  } );

		self->update();
		paging = self->client;

		i += self->batch_size;
		return true;
	}
}

VM_END_MODULE()
