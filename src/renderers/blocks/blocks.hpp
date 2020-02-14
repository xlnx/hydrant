#pragma once

#include <hydrant/renderer.hpp>
#include <hydrant/const_texture_3d.hpp>
#include <hydrant/cuda_image.hpp>
#include "blocks_shader.hpp"

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	struct BlocksRendererConfig : vm::json::Serializable<BlocksRendererConfig>
	{
		VM_JSON_FIELD( std::string, mode ) = "volume";
		VM_JSON_FIELD( float, density ) = 1e-2f;
	};

	struct BlocksRenderer : Renderer
	{
		using Shader = BlocksShader;
		using Super = Renderer;

		virtual bool init( std::shared_ptr<Dataset> const &dataset,
						   RendererConfig const &cfg ) override;

		virtual void offline_render( std::string const &dst_path,
									 Camera const &camera ) override;

	private:
		cufx::Device device = cufx::Device::scan()[ 0 ];
		Shader shader;
		Exhibit exhibit;
		vm::Option<CudaImage<typename Shader::Pixel>> image;
		vm::Option<ConstTexture3D<float>> chebyshev;
		vm::Option<ConstTexture3D<float>> mean;
	};
}

VM_END_MODULE()
