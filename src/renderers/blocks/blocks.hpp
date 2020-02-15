#pragma once

#include <hydrant/renderer.hpp>
#include <hydrant/const_texture_3d.hpp>
#include <hydrant/basic_renderer.hpp>
#include <hydrant/image.hpp>
#include "blocks_shader.hpp"

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	struct BlocksRendererConfig : vm::json::Serializable<BlocksRendererConfig>
	{
		VM_JSON_FIELD( BlocksRenderMode, mode ) = BlocksRenderMode::Volume;
		VM_JSON_FIELD( float, density ) = 1e-2f;
	};

	struct BlocksRenderer : BasicRenderer
	{
		using Shader = BlocksShader;
		using Super = BasicRenderer;

		virtual bool init( std::shared_ptr<Dataset> const &dataset,
						   RendererConfig const &cfg ) override;

		virtual void offline_render( std::string const &dst_path,
									 Camera const &camera ) override;

	private:
		Shader shader;
		Exhibit exhibit;
		Image<typename Shader::Pixel> image;
		ThumbnailTexture<int> chebyshev;
		ThumbnailTexture<float> mean;
	};
}

VM_END_MODULE()
