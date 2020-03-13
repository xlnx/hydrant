#pragma once

#include <varch/utils/io.hpp>
#include <hydrant/basic_renderer.hpp>
#include <hydrant/transfer_fn.hpp>
#include "isosurface_shader.hpp"

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	struct IsosurfaceRendererParams : vm::json::Serializable<IsosurfaceRendererParams>
	{
		VM_JSON_FIELD( IsosurfaceRenderMode, mode ) = IsosurfaceRenderMode::Color;
		VM_JSON_FIELD( vec3, surface_color ) = { 1.f, 1.f, 1.f };
		VM_JSON_FIELD( float, isovalue ) = 0.5f;
		VM_JSON_FIELD( std::size_t, mem_limit_mb ) = 1024 * 2;
	};

	struct IsosurfaceRenderer : BasicRenderer<IsosurfaceShader>
	{
		using Super = BasicRenderer<IsosurfaceShader>;

		bool init( std::shared_ptr<Dataset> const &dataset,
				   RendererConfig const &cfg ) override;

		void update( vm::json::Any const &params ) override;

	protected:
		OfflineRenderCtx *create_offline_render_ctx() override;

		cufx::Image<> offline_render_ctxed( OfflineRenderCtx &ctx, Camera const &camera ) override;

		void realtime_render_dynamic( IRenderLoop &loop ) override;

	private:
		vol::MtArchive *lvl0_arch;

		std::shared_ptr<vol::Thumbnail<int>> chebyshev_thumb;
		ThumbnailTexture<int> chebyshev;
	};
}

VM_END_MODULE()
