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
		lk.unlock();
		pipeline.cv.notify_one();
	}

	IUnarchivePipeline::IUnarchivePipeline( Unarchiver & unarchiver,
											UnarchivePipelineOptions const &opts ) :
	  unarchiver( unarchiver ),
	  device( opts.device )
	{
		auto pad_bs = unarchiver.padded_block_size();
		if ( opts.device.has_value() ) {
			buf.reset( new GlobalBuffer3D<unsigned char>( uvec3( pad_bs ),
														  opts.device.value() ) );
		} else {
			buf.reset( new HostBuffer3D<unsigned char>( uvec3( pad_bs ) ) );
		}
	}

	void IUnarchivePipeline::start()
	{
		curr_batch_size = 0;
		required.resize( 0 );
		should_stop = false;
		worker.reset( new cufx::WorkerThread( [&] { this->run(); }, device ) );
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
				top_k = lk.top_k_idxs( 16 );
			}
			size_t nbytes = 0;
			unarchiver.unarchive(
			  top_k,
			  [&]( Idx const &idx, VoxelStreamPacket const &pkt ) {
				  pkt.append_to( buf->view_1d() );
				  nbytes += pkt.length;
				  if ( nbytes >= buf->bytes() ) {
					  {
						  auto lk = this->lock();
						  auto it = find_if(
							required.begin(), required.end(),
							[&]( auto &a ) {
								return a.first == idx;
							} );
						  if ( it != required.end() ) {
							  required.erase( it );
						  }
					  }
					  on_data( idx, *buf );
					  nbytes = 0;
				  }
			  } );
		}
	}
}

VM_END_MODULE()
