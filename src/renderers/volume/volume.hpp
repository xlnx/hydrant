#pragma once

#include <varch/utils/io.hpp>
#include <hydrant/unarchiver.hpp>
#include <hydrant/core/renderer.hpp>
#include <hydrant/bridge/const_texture_3d.hpp>
#include <hydrant/bridge/image.hpp>
#include <hydrant/buffer3d.hpp>
#include "volume_shader.hpp"

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	struct VolumeRendererConfig : vm::json::Serializable<VolumeRendererConfig>
	{
		VM_JSON_FIELD( ShadingDevice, device ) = ShadingDevice::Cuda;
	};

	struct VolumeRenderer : IRenderer
	{
		using Shader = VolumeShader;
		using Super = IRenderer;

		virtual bool init( std::shared_ptr<Dataset> const &dataset,
						   RendererConfig const &cfg ) override;

		virtual void offline_render( std::string const &dst_path,
									 Camera const &camera ) override;

	private:
		vm::Option<cufx::Device> device;
		Shader shader;
		Exhibit exhibit;

		std::shared_ptr<Unarchiver> uu;

		Image<typename Shader::Pixel> image;
		ConstTexture3D<int> chebyshev;
		ConstTexture3D<int> present;

		vm::Option<Buffer3D<int>> present_buf;

		std::vector<vol::Idx> block_idxs;
		std::vector<glm::vec3> block_ccs;
		std::vector<int> pidx;
	};
}

VM_END_MODULE()
