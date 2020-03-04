#pragma once

#include <varch/utils/io.hpp>
#include <hydrant/basic_renderer.hpp>
#include <hydrant/transfer_fn.hpp>
#include "paging_shader.hpp"

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	struct PagingRendererConfig : vm::json::Serializable<PagingRendererConfig>
	{
		// VM_JSON_FIELD( IsosurfaceRenderMode, mode ) = IsosurfaceRenderMode::Color;
		VM_JSON_FIELD( std::size_t, mem_limit_mb ) = 1024 * 2;
	};

	struct PagingRenderer : BasicRenderer<PagingShader>
	{
		using Super = BasicRenderer<PagingShader>;

		bool init( std::shared_ptr<Dataset> const &dataset,
				   RendererConfig const &cfg ) override;

		cufx::Image<> offline_render_ctxed( OfflineRenderCtx &ctx, Camera const &camera ) override;

		void realtime_render_dynamic( IRenderLoop &loop ) override;

	private:
		vol::MtArchive *lvl0_arch;

		std::shared_ptr<vol::Thumbnail<int>> chebyshev_thumb;
		ThumbnailTexture<int> chebyshev;
	};
}

VM_END_MODULE()
