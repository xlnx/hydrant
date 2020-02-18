#pragma once

#include <varch/utils/io.hpp>
#include <hydrant/unarchiver.hpp>
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
	};

	struct VolumeRenderer : BasicRenderer<VolumeShader>
	{
		using Super = BasicRenderer<VolumeShader>;

		virtual bool init( std::shared_ptr<Dataset> const &dataset,
						   RendererConfig const &cfg ) override;

		virtual cufx::Image<> offline_render( Camera const &camera ) override;

	private:
		std::shared_ptr<Unarchiver> uu;

		TransferFn transfer_fn;

		ThumbnailTexture<int> chebyshev;

		Texture3D<int> present;
		HostBuffer3D<int> present_buf;

		std::vector<Texture3D<unsigned char>> cache;

		std::vector<vol::Idx> block_idxs;
		std::vector<glm::vec3> block_ccs;
		std::vector<int> pidx;
	};
}

VM_END_MODULE()
