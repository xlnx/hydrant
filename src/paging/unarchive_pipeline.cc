#include <hydrant/paging/unarchive_pipeline.hpp>

VM_BEGIN_MODULE( hydrant )

using namespace std;
using namespace vol;

VM_EXPORT
{
	vector<Idx> IUnarchivePipeline::Lock::top_k_idxs( size_t k )
	{
		pipeline.curr_batch_size = std::min( k, pipeline.required.size() );
		auto seq_beg = pipeline.required.begin();
		auto seq_end = seq_beg + pipeline.curr_batch_size;
		std::nth_element(
		  seq_beg, seq_end, pipeline.required.end(),
		  []( auto &a, auto &b ) {
			  return a.second < b.second;
		  } );
		std::vector<vol::Idx> res( pipeline.curr_batch_size );
		std::transform(
		  seq_beg, seq_end, res.begin(),
		  []( auto &a ) {
			  return a.first;
		  } );
		/* init still need */
		pipeline.still_need.reset( new StillNeed[ pipeline.curr_batch_size ] );
		for ( int i = 0; i != pipeline.curr_batch_size; ++i ) {
			pipeline.still_need[ i ].first.store( true );
			pipeline.still_need[ i ].second = res[ i ];
		}
		sort_required_idxs();
		return res;
	}

	void IUnarchivePipeline::Lock::sort_required_idxs()
	{
		std::sort(
		  pipeline.required.begin(), pipeline.required.end(),
		  []( auto &a, auto &b ) {
			  return a.first < b.first;
		  } );
	}

	void IUnarchivePipeline::Lock::require_impl( bool is_ordered )
	{
		if ( !is_ordered ) { sort_required_idxs(); }
		/* update still need */
		if ( pipeline.still_need ) {
			for ( int i = 0; i != pipeline.curr_batch_size; ++i ) {
				auto &need = pipeline.still_need[ i ];
				if ( need.first.load() ) {
					auto pp = std::make_pair( need.second, 0.f );
					auto sres = std::binary_search(
					  pipeline.required.begin(), pipeline.required.end(), pp,
					  []( auto &a, auto &b ) {
						  return a.first == b.first;
					  } );
					if ( !sres ) {
						need.first.store( false );
					}
				}
			}
		}
		lk.unlock();
		pipeline.cv.notify_one();
	}

	IUnarchivePipeline::IUnarchivePipeline( Unarchiver & unarchiver,
											UnarchivePipelineOptions const &opts ) :
	  unarchiver( unarchiver )
	{
		auto pad_bs = unarchiver.padded_block_size();
		if ( opts.device.has_value() ) {
			buf.reset( new GlobalBuffer3D<unsigned char>( uvec3( pad_bs ), opts.device.value() ) );
		} else {
			buf.reset( new HostBuffer3D<unsigned char>( uvec3( pad_bs ) ) );
		}
	}

	void IUnarchivePipeline::start()
	{
		still_need.reset();
		curr_batch_size = 0;
		required.resize( 0 );
		should_stop = false;
		worker.reset( new std::thread( [&] { this->run(); } ) );
	}

	void IUnarchivePipeline::stop()
	{
		should_stop = true;
		cv.notify_one();
		worker->join();
		worker.reset();
	}

	void IUnarchivePipeline::run()
	{
		while ( !should_stop ) {
			vector<Idx> top_k;
			{
				auto lk = this->lock();
				cv.wait( lk.lk, [&] { return should_stop || required.size(); } );
				if ( should_stop ) { return; }
				top_k = lk.top_k_idxs( 4 );
			}
			map<Idx, int> idx_lookup;
			for ( int i = 0; i != top_k.size(); ++i ) {
				idx_lookup[ top_k[ i ] ] = i;
			}
			size_t nbytes = 0;
			unarchiver.unarchive(
			  top_k,
			  [&]( Idx const &idx, VoxelStreamPacket const &pkt ) {
				  auto sn_idx = idx_lookup[ idx ];
				  if ( still_need[ sn_idx ].first.load() ) {
					  pkt.append_to( buf->view_1d() );
				  }
				  nbytes += pkt.length;
				  if ( nbytes >= buf->bytes() ) {
					  {
						  auto lk = this->lock();
						  auto it = find_if(
							required.begin(), required.end(),
							[&]( auto &a ) {
								return a.first == idx;
							} );
						  required.erase( it );
					  }
					  if ( still_need[ sn_idx ].first.load() ) {
						  on_data( idx, *buf );
					  }
					  //   auto fut = cache[ blkid ].source( buf->view_3d() );
					  //   fut.wait();
					  //   vaddr_buf[ glm::vec3( idx.x, idx.y, idx.z ) ] = blkid;
					  nbytes = 0;
				  }
			  } );
		}
	}
}

VM_END_MODULE()
