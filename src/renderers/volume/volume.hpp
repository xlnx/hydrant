#pragma once

#include <varch/utils/io.hpp>
#include <hydrant/bridge/buffer3d.hpp>
#include <hydrant/basic_renderer.hpp>
#include <hydrant/transfer_fn.hpp>
#include "volume_shader.hpp"

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	struct VolumeRendererConfig : vm::json::Serializable<VolumeRendererConfig>
	{
		VM_JSON_FIELD( TransferFnConfig, transfer_fn );
		VM_JSON_FIELD( float, density ) = 1e-2f;
		VM_JSON_FIELD( std::size_t, mem_limit_mb ) = 1024 * 2;
		VM_JSON_FIELD( VolumeRenderMode, mode ) = VolumeRenderMode::Volume;
	};

	struct VolumeRenderer : BasicRenderer<VolumeShader>
	{
		using Super = BasicRenderer<VolumeShader>;

		bool init( std::shared_ptr<Dataset> const &dataset,
				   RendererConfig const &cfg ) override;

		cufx::Image<> offline_render( Camera const &camera ) override;

		void render_loop( IRenderLoop &loop ) override;

	private:
		TransferFn transfer_fn;
		vol::MtArchive *lvl0_arch;

		std::shared_ptr<vol::Thumbnail<int>> chebyshev_thumb;
		ThumbnailTexture<int> chebyshev;
	};
}

VM_END_MODULE()
