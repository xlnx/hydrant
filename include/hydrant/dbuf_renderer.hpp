#pragma once

#include <varch/thumbnail.hpp>
#include <hydrant/basic_renderer.hpp>
#include <hydrant/double_buffering.hpp>
#include <hydrant/octree_culler.hpp>

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	struct DbufRtRenderCtx : vm::Dynamic
	{
	};
	
	template <typename Shader>
	struct DbufRenderer : BasicRenderer<Shader>
	{
		void realtime_render_dynamic( IRenderLoop &loop, MpiComm const &comm )
		{
			std::unique_ptr<DbufRtRenderCtx> ctx( create_dbuf_rt_render_ctx() );
			OctreeCuller culler( this->exhibit, this->chebyshev_thumb );
				
			FnDoubleBuffering loop_drv(
			  ImageOptions{}
			    .set_device( this->device )
			    .set_resolution( this->resolution ),
			  loop,
			  [&]( auto &frame, auto frame_idx ) {
				  auto bbox = slice_bbox( loop.camera, comm );
				  culler.set_bbox( bbox );
				  this->shader.bbox = Box3D{ bbox.min, bbox.max };
				  dbuf_rt_render_frame( frame, *ctx, loop, culler, comm );
			  },
			  [&]( auto &frame, auto frame_idx ) {
				  auto fp = frame.fetch_data();
				  loop.on_frame( fp );
			  });

			loop_drv.run();
		}

	private:
		BoundingBox slice_bbox( Camera const &camera, MpiComm const &comm )
		{
			static vec3 axis_map[] = {
				{ 0, 0, 1 }, { 0, 0, -1 },
				{ 1, 0, 0 }, { -1, 0, 0 },
				{ 0, 1, 0 }, { 0, -1, 0 }
			};
			auto dist = camera.target - camera.position;
			int max_idx = -1;
			float max_dt = -INFINITY;
			for ( int i = 0; i != 6; ++i ) {
				auto dt = dot( dist, axis_map[ i ] );
				if ( dt > max_dt ) {
					max_idx = i;
					max_dt = dt;
				}
			}
			ivec3 axis = axis_map[ max_idx ];
			auto naxis = abs( axis );
			auto idim = ivec3( this->dim );
			auto nslices = idim.x * naxis.x + idim.y * naxis.y + idim.z * naxis.z;
			auto my_slice = deploy_slice( nslices, comm );
			auto plane = idim * ( 1 - naxis );
			auto orig = ( axis.x + axis.y + axis.z < 0 ) ? naxis * nslices : ivec3( 0 );
			auto p0 = orig + axis * my_slice.first;
			auto p1 = orig + plane + axis * my_slice.second;
			return BoundingBox{}
			             .set_min( min( p0, p1 ) )
				         .set_max( max( p0, p1 ) );
		}

		std::pair<int, int> deploy_slice( int nslices, MpiComm const &comm )
		{
			// at most one slice per slave
			if ( nslices <= comm.size ) {
				if ( comm.rank >= nslices ) {
					return std::make_pair( 0, 0 );
				} else {
					return std::make_pair( comm.rank, comm.rank + 1 );
				}
			} else {
				// implement balanced deployment strategy
				int n = nslices / comm.size;
				if ( comm.rank == comm.size - 1 ) {
					return std::make_pair( comm.rank * n, nslices );
				} else {
					return std::make_pair( comm.rank * n, comm.rank * n + n );
				}
			}
		}
		
	public:
		virtual DbufRtRenderCtx *create_dbuf_rt_render_ctx()
		{
			return new DbufRtRenderCtx;
		}

		virtual void dbuf_rt_render_frame( Image<cufx::StdByte3Pixel> &frame,
										   DbufRtRenderCtx &ctx,
										   IRenderLoop &loop,
										   OctreeCuller &culler,
										   MpiComm const &comm ) = 0;

	protected:
		std::shared_ptr<vol::Thumbnail<int>> chebyshev_thumb;
	};
}

VM_END_MODULE()
