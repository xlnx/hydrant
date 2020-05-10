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
#include <hydrant/bridge/buffer_3d.hpp>
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
			void require( std::vector<vol::Idx> const &missing ) &&;
				
		private:
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
		std::vector<vol::Idx> required;
		vm::Option<cufx::Device> device;
		std::size_t curr_batch_size = 0;
		std::unique_ptr<IBuffer3D<unsigned char>> buf;
		std::unique_ptr<cufx::WorkerThread> worker;
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
