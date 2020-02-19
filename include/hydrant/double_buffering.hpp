#pragma once

#include <array>
#include <hydrant/core/render_loop.hpp>
#include <hydrant/bridge/image.hpp>

#include <thread>
#include <mutex>
#include <condition_variable>

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	struct IDoubleBuffering : vm::NoCopy, vm::NoMove, vm::Dynamic
	{
		IDoubleBuffering( ImageOptions const &opts, IRenderLoop &loop ) :
		  frames{ opts, opts },
		  loop( loop )
		{
		}

	public:
		virtual void render( Image<cufx::StdByte4Pixel> &frame_out ) = 0;

		void run()
		{
			int prod_frame_idx = 0, cons_frame_idx = 1;
			bool should_continue = true;
			std::mutex mut;
			std::condition_variable cv;

			loop.post_loop();

			std::thread _( [&] {
				while ( should_continue ) {
					std::unique_lock<std::mutex> lk( mut );
					cv.wait( lk, [&] { return !( should_continue = !loop.should_stop() ) ||
											  prod_frame_idx != cons_frame_idx; } );
					if ( !should_continue ) {
						lk.unlock();
						cv.notify_one();
						return;
					}
					loop.post_frame();
					render( frames[ prod_frame_idx ] );
					prod_frame_idx = 1 - prod_frame_idx;
					lk.unlock();
					cv.notify_one();
				}
			} );

			while ( should_continue = !loop.should_stop() ) {
				std::unique_lock<std::mutex> lk( mut );
				cv.wait( lk, [&] { return !( should_continue = !loop.should_stop() ) ||
										  prod_frame_idx == cons_frame_idx; } );
				if ( !should_continue ) {
					lk.unlock();
					cv.notify_one();
					break;
				}
				cons_frame_idx = 1 - cons_frame_idx;
				auto fp = frames[ cons_frame_idx ].fetch_data();
				loop.on_frame( fp );
				lk.unlock();
				cv.notify_one();
			}

			cv.notify_one();
			_.join();

			loop.after_loop();
		}

	private:
		std::array<Image<cufx::StdByte4Pixel>, 2> frames;
		IRenderLoop &loop;
	};

	struct FnDoubleBuffering : IDoubleBuffering
	{
		FnDoubleBuffering( ImageOptions const &opts, IRenderLoop &loop,
						   std::function<void( Image<cufx::StdByte4Pixel> & )> const &render_fn ) :
		  IDoubleBuffering( opts, loop ),
		  render_fn( render_fn )
		{
		}

	public:
		void render( Image<cufx::StdByte4Pixel> &frame_out ) override { render_fn( frame_out ); }

	private:
		std::function<void( Image<cufx::StdByte4Pixel> & )> render_fn;
	};
}

VM_END_MODULE()
