#pragma once

#include <set>
#include <thread>
#include <mutex>
#include <memory>
#include <atomic>
#include <algorithm>
#include <condition_variable>
#include <VMUtils/option.hpp>
#include <cudafx/device.hpp>
#include <hydrant/bridge/buffer3d.hpp>
#include <hydrant/unarchiver.hpp>

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	struct UnarchivePipelineOptions
	{
		VM_DEFINE_ATTRIBUTE( std::size_t, batch_size ) = 4;
		VM_DEFINE_ATTRIBUTE( vm::Option<cufx::Device>, device ) = vm::None{};
	};

	struct IUnarchivePipeline : vm::NoCopy, vm::NoMove
	{
		IUnarchivePipeline( Unarchiver &unarchiver,
							UnarchivePipelineOptions const &opts = UnarchivePipelineOptions{} );

	public:
		struct Lock;
		friend struct Lock;

		struct Lock : vm::NoCopy
		{
		private:
			Lock( IUnarchivePipeline &pipeline ) :
			  lk( pipeline.mut ),
			  pipeline( pipeline )
			{
			}

		public:
			template <typename InputIterator, typename PriorityOp>
			void require( InputIterator const &seq_beg, InputIterator const &seq_end,
						  PriorityOp const &priority_op, bool is_ordered = false ) &&
			{
				pipeline.required.resize( seq_end - seq_beg );
				std::transform(
				  seq_beg, seq_end, pipeline.required.begin(),
				  [&]( vol::Idx const &idx ) {
					  return std::make_pair( idx, priority_op( idx ) );
				  } );
				require_impl( is_ordered );
			}

		private:
			void sort_required_idxs();

			void require_impl( bool is_ordered );

			std::vector<vol::Idx> top_k_idxs( std::size_t k );

		private:
			std::unique_lock<std::mutex> lk;
			IUnarchivePipeline &pipeline;
			friend struct IUnarchivePipeline;
		};

	public:
		void start();

		void stop();

		Lock lock() { return *this; }

	public:
		virtual void on_data( vol::Idx const &, IBuffer3D<unsigned char> & ) = 0;

	private:
		void run();

	private:
		Unarchiver &unarchiver;
		bool should_stop = false;
		std::mutex mut;
		std::condition_variable cv;
		std::vector<std::pair<vol::Idx, float>> required;
		using StillNeed = std::pair<std::atomic_bool, vol::Idx>;
		std::unique_ptr<StillNeed[]> still_need;
		std::size_t curr_batch_size = 0;
		std::unique_ptr<IBuffer3D<unsigned char>> buf;
		std::unique_ptr<std::thread> worker;
	};

	struct FnUnarchivePipeline : IUnarchivePipeline
	{
		using OnDataFn = std::function<void( vol::Idx const &, IBuffer3D<unsigned char> & )>;

	public:
		FnUnarchivePipeline( Unarchiver &unarchiver,
							 OnDataFn const &on_data_fn,
							 UnarchivePipelineOptions const &opts = UnarchivePipelineOptions{} ) :
		  IUnarchivePipeline( unarchiver, opts ),
		  on_data_fn( on_data_fn )
		{
		}

	public:
		void on_data( vol::Idx const &idx, IBuffer3D<unsigned char> &buffer ) override
		{
			on_data_fn( idx, buffer );
		}

	private:
		OnDataFn on_data_fn;
	};
}

VM_END_MODULE()
