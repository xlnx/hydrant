#pragma once

#include <array>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <hydrant/core/render_loop.hpp>
#include <hydrant/bridge/image.hpp>

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
		virtual void render( Image<cufx::StdByte4Pixel> &frame_out, int frame_idx ) = 0;

		virtual void display( Image<cufx::StdByte4Pixel> &frame_in, int frame_idx ) = 0;

		void run();

	private:
		std::array<Image<cufx::StdByte4Pixel>, 2> frames;
		IRenderLoop &loop;
	};

	struct FnDoubleBuffering : IDoubleBuffering
	{
		using FnType = std::function<void( Image<cufx::StdByte4Pixel> &, int )>;

		FnDoubleBuffering( ImageOptions const &opts, IRenderLoop &loop,
						   FnType const &render_fn,
						   FnType const &display_fn ) :
		  IDoubleBuffering( opts, loop ),
		  render_fn( render_fn ),
		  display_fn( display_fn )
		{
		}

	public:
		void render( Image<cufx::StdByte4Pixel> &frame_out, int frame_idx ) override
		{
			render_fn( frame_out, frame_idx );
		}

		void display( Image<cufx::StdByte4Pixel> &frame_in, int frame_idx ) override
		{
			display_fn( frame_in, frame_idx );
		}

	private:
		FnType render_fn, display_fn;
	};
}

VM_END_MODULE()
