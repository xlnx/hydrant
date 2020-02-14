#pragma once

#include <varch/utils/io.hpp>
#include <hydrant/unarchiver.hpp>
#include <hydrant/renderer.hpp>
#include <hydrant/const_texture_3d.hpp>
#include <hydrant/cuda_image.hpp>
#include <hydrant/core/buffer3d.hpp>
#include "volume_shader.hpp"

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	struct VolumeRendererConfig : vm::json::Serializable<VolumeRendererConfig>
	{
	};

	struct VolumeRenderer : Renderer
	{
		using Shader = VolumeShader;
		using Super = Renderer;

		virtual bool init( std::shared_ptr<Dataset> const &dataset,
						   RendererConfig const &cfg ) override;

		virtual void offline_render( std::string const &dst_path,
									 Camera const &camera ) override;

	private:
		cufx::Device device = cufx::Device::scan()[ 0 ];
		Shader shader;
		Exhibit exhibit;

		std::shared_ptr<Unarchiver> uu;

		vm::Option<CudaImage<typename Shader::Pixel>> image;
		vm::Option<ConstTexture3D<float>> chebyshev;
		vm::Option<ConstTexture3D<float>> mean;
		vm::Option<ConstTexture3D<int>> present;

		vm::Option<Buffer3D<int>> present_buf;

		std::vector<vol::Idx> block_idxs;
		std::vector<glm::vec3> block_ccs;
		std::vector<int> pidx;
	};
}

VM_END_MODULE()
