#include <varch/utils/io.hpp>
#include <varch/unarchive/unarchiver.hpp>
#include <hydrant/basic_renderer.hpp>
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

		chebyshev = load_thumbnail<int>(
		  dataset->root.resolve( lvl0_arch.thumbnails[ "chebyshev" ] ).resolved() );
		shader.chebyshev_tex = chebyshev.get();

		mean = load_thumbnail<float>(
		  dataset->root.resolve( lvl0_arch.thumbnails[ "mean" ] ).resolved() );
		shader.mean_tex = mean.get();

		return true;
	}

	void BlocksRenderer::offline_render( std::string const &dst_path,
										 Camera const &camera )
	{
		raycaster.cast_cpu( exhibit, camera, film.view(), shader );
		film.fetch_data().dump( dst_path );
	}

	REGISTER_RENDERER( BlocksRenderer, "Blocks" );
}

VM_END_MODULE()
