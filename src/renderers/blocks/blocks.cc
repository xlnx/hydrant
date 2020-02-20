#include <varch/utils/io.hpp>
#include <VMUtils/timer.hpp>
#include <varch/unarchive/unarchiver.hpp>
#include <hydrant/basic_renderer.hpp>
#include <hydrant/core/render_loop.hpp>
#include <hydrant/octree_culler.hpp>
#include "blocks.hpp"

using namespace std;
using namespace vol;

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	bool BlocksRenderer::init( std::shared_ptr<Dataset> const &dataset,
							   RendererConfig const &cfg )
	{
		if ( !Super::init( dataset, cfg ) ) { return false; }

		auto params = cfg.params.get<BlocksRendererParams>();
		shader.render_mode = params.mode;
		shader.density = params.density;

		auto &lvl0_arch = dataset->meta.sample_levels[ 0 ].archives[ 0 ];

		auto chebyshev_thumb = std::make_shared<vol::Thumbnail<int>>(
		  dataset->root.resolve( lvl0_arch.thumbnails[ "chebyshev" ] ).resolved() );
		chebyshev = create_texture( chebyshev_thumb );
		shader.chebyshev = chebyshev.sampler();

		auto mean_thumb = std::make_shared<vol::Thumbnail<float>>(
		  dataset->root.resolve( lvl0_arch.thumbnails[ "mean" ] ).resolved() );
		mean = create_texture( mean_thumb );
		shader.mean_tex = mean.sampler();

		return true;
	}

	cufx::Image<> BlocksRenderer::offline_render( Camera const &camera )
	{
		auto film = create_film();
		raycaster.cast( exhibit,
						camera,
						film.view(),
						shader,
						RaycastingOptions{}
						  .set_device( device ) );
		return film.fetch_data().dump();
	}

	void BlocksRenderer::render_loop( IRenderLoop & loop )
	{
		loop.post_loop();
		while ( !loop.should_stop() ) {
			loop.post_frame();
			OctreeCuller culler( exhibit, dim );
			{
				vm::Timer::Scoped tt( []( auto dt ) {
					vm::println( "{}", dt.ms() );
				} );
				auto idxs = culler.cull( loop.camera );
				static int rnd = 0;
				if ( !( rnd = ( rnd + 1 ) % 60 ) ) {
					vm::println( "{}", idxs );
				}
			}
			auto frame = offline_render( loop.camera );
			loop.on_frame( frame );
		}
		loop.after_loop();
	}

	REGISTER_RENDERER( BlocksRenderer, "Blocks" );
}

VM_END_MODULE()
