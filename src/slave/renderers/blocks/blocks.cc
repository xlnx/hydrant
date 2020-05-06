#include <varch/utils/io.hpp>
#include <VMUtils/timer.hpp>
#include <varch/unarchive/unarchiver.hpp>
#include <hydrant/basic_renderer.hpp>
#include <hydrant/core/render_loop.hpp>
#include <hydrant/octree_culler.hpp>
#include <hydrant/basic_renderer.hpp>
#include "blocks_shader.hpp"

using namespace std;
using namespace vol;

struct BlocksRenderer : BasicRenderer<BlocksShader>
{
	using Super = BasicRenderer<BlocksShader>;

	bool init( std::shared_ptr<Dataset> const &dataset,
			   RendererConfig const &cfg ) override;

	void update( vm::json::Any const &params ) override;

	cufx::Image<> offline_render_ctxed( OfflineRenderCtx &ctx,
										Camera const &camera ) override;

private:
	ThumbnailTexture<int> chebyshev;
	ThumbnailTexture<float> mean;
};

bool BlocksRenderer::init( std::shared_ptr<Dataset> const &dataset,
						   RendererConfig const &cfg )
{
	if ( !Super::init( dataset, cfg ) ) { return false; }

	auto &lvl0_arch = dataset->meta.sample_levels[ 0 ].archives[ 0 ];

	auto chebyshev_thumb = std::make_shared<vol::Thumbnail<int>>(
	  dataset->root.resolve( lvl0_arch.thumbnails[ "chebyshev" ] ).resolved() );
	chebyshev = create_texture( chebyshev_thumb );
	shader.chebyshev = chebyshev.sampler();

	auto mean_thumb = std::make_shared<vol::Thumbnail<float>>(
	  dataset->root.resolve( lvl0_arch.thumbnails[ "mean" ] ).resolved() );
	mean = create_texture( mean_thumb );
	shader.mean_tex = mean.sampler();

	update( cfg.params );

	return true;
}

void BlocksRenderer::update( vm::json::Any const &params_in )
{
	Super::update( params_in );

	auto params = params_in.get<BlocksRendererParams>();
	shader.render_mode = params.mode;
	shader.density = params.density;
}

cufx::Image<> BlocksRenderer::offline_render_ctxed( OfflineRenderCtx &ctx, Camera const &camera )
{
	auto film = create_film();
	raycaster.ray_emit_pass( exhibit,
							 camera,
							 film.view(),
							 shader,
							 RaycastingOptions{}
							   .set_device( device ) );
	film.update_device_view();
	return film.fetch_data().dump();
}

REGISTER_RENDERER( BlocksRenderer, "Blocks" );
