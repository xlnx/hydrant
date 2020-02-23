#pragma once

#include <varch/utils/io.hpp>
#include <hydrant/basic_renderer.hpp>
#include <hydrant/transfer_fn.hpp>
#include "isosurface_shader.hpp"

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	struct IsosurfaceRendererConfig : vm::json::Serializable<IsosurfaceRendererConfig>
	{
		VM_JSON_FIELD( IsosurfaceRenderMode, mode ) = IsosurfaceRenderMode::Color;
		VM_JSON_FIELD( float, isovalue ) = 0.5f;
		VM_JSON_FIELD( std::size_t, mem_limit_mb ) = 1024 * 2;
	};

	struct IsosurfaceRenderer : BasicRenderer<IsosurfaceShader>
	{
		using Super = BasicRenderer<IsosurfaceShader>;

		bool init( std::shared_ptr<Dataset> const &dataset,
				   RendererConfig const &cfg ) override;

		cufx::Image<> offline_render( Camera const &camera ) override;

		void render_loop( IRenderLoop &loop ) override;

	private:
		vol::MtArchive *lvl0_arch;

		std::shared_ptr<vol::Thumbnail<int>> chebyshev_thumb;
		ThumbnailTexture<int> chebyshev;
	};
}

VM_END_MODULE()
