#pragma once

#include <hydrant/basic_renderer.hpp>
#include "blocks_shader.hpp"

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	struct BlocksRendererParams : vm::json::Serializable<BlocksRendererParams>
	{
		VM_JSON_FIELD( BlocksRenderMode, mode ) = BlocksRenderMode::Volume;
		VM_JSON_FIELD( float, density ) = 1e-2f;
	};

	struct BlocksRenderer : BasicRenderer<BlocksShader>
	{
		using Super = BasicRenderer<BlocksShader>;

		bool init( std::shared_ptr<Dataset> const &dataset,
				   RendererConfig const &cfg ) override;

		void update( vm::json::Any const &params ) override;

		cufx::Image<> offline_render_ctxed( OfflineRenderCtx &ctx, Camera const &camera ) override;

	private:
		ThumbnailTexture<int> chebyshev;
		ThumbnailTexture<float> mean;
	};
}

VM_END_MODULE()
