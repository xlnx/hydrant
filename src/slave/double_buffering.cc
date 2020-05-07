#include <hydrant/double_buffering.hpp>

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	void IDoubleBuffering::run()
	{
		int prod_frame_idx = 0, cons_frame_idx = 1;
		bool should_continue = true;
		std::mutex mut;
		std::condition_variable cv;

		loop.post_loop();

		cufx::WorkerThread _( [&] {
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
				render( frames[ prod_frame_idx ], prod_frame_idx );
				prod_frame_idx = 1 - prod_frame_idx;
				lk.unlock();
				cv.notify_one();
			}
		}, device );

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
			display( frames[ cons_frame_idx ], cons_frame_idx );
			loop.after_frame();
			lk.unlock();
			cv.notify_one();
		}

		cv.notify_one();
		_.join();

		loop.after_loop();
	}
}

VM_END_MODULE()
