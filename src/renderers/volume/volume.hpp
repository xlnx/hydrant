
#include <hydrant/basic_renderer.hpp>
#include <hydrant/transfer_fn.hpp>
#include "volume_shader.hpp"

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	struct VolumeRendererParams : vm::json::Serializable<VolumeRendererParams>
	{
		VM_JSON_FIELD( TransferFnConfig, transfer_fn );
		VM_JSON_FIELD( float, density ) = 1e-2f;
		VM_JSON_FIELD( std::size_t, mem_limit_mb ) = 1024 * 2;
	};

	struct VolumeRenderer : BasicRenderer<VolumeShader>
	{
		using Super = BasicRenderer<VolumeShader>;

		bool init( std::shared_ptr<Dataset> const &dataset,
				   RendererConfig const &cfg ) override;

		void update( vm::json::Any const &params_in ) override;

	protected:
		OfflineRenderCtx *create_offline_render_ctx() override;

		cufx::Image<> offline_render_ctxed( OfflineRenderCtx &ctx, Camera const &camera ) override;

		void realtime_render_dynamic( IRenderLoop &loop ) override;

	private:
		TransferFn transfer_fn;
		vol::MtArchive *lvl0_arch;

		std::shared_ptr<vol::Thumbnail<int>> chebyshev_thumb;
		ThumbnailTexture<int> chebyshev;
	};
}

VM_END_MODULE()
