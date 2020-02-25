#pragma once

#include <varch/thumbnail.hpp>
#include <cudafx/device.hpp>
#include <hydrant/bridge/image.hpp>
#include <hydrant/core/renderer.hpp>
#include <hydrant/bridge/texture_3d.hpp>
#include <hydrant/core/render_loop.hpp>

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	template <typename Shader>
	struct BasicRenderer;

	template <typename T>
	struct ThumbnailTexture : Texture3D<T>
	{
		using Texture3D<T>::Texture3D;

	private:
		std::shared_ptr<vol::Thumbnail<T>> thumb;
		template <typename Shader>
		friend struct BasicRenderer;
	};

	struct BasicRendererParams : vm::json::Serializable<BasicRendererParams>
	{
		VM_JSON_FIELD( ShadingDevice, device ) = ShadingDevice::Cuda;
		VM_JSON_FIELD( int, max_steps ) = 500;
		VM_JSON_FIELD( vec3, clear_color ) = vec3( 0 );
	};

	template <typename Shader>
	struct BasicRenderer : IRenderer
	{
		bool init( std::shared_ptr<Dataset> const &dataset,
				   RendererConfig const &cfg ) override
		{
			if ( !IRenderer::init( dataset, cfg ) ) { return false; }

			auto params = cfg.params.get<BasicRendererParams>();
			if ( params.device == ShadingDevice::Cuda ) {
				device = cufx::Device::get_default();
				if ( !device.has_value() ) {
					vm::println( "cuda device not found, fallback to cpu render mode" );
				}
			}
			shader.max_steps = params.max_steps;
			clear_color = params.clear_color;

			auto &lvl0 = dataset->meta.sample_levels[ 0 ];
			auto &lvl0_arch = lvl0.archives[ 0 ];
			auto lvl0_blksz = float( lvl0_arch.block_size );
			dim = vec3( lvl0_arch.dim.x,
						lvl0_arch.dim.y,
						lvl0_arch.dim.z );
			auto raw = vec3( lvl0.raw.x,
							 lvl0.raw.y,
							 lvl0.raw.z );
			auto f_dim = raw / lvl0_blksz;
			exhibit = Exhibit{}
						.set_center( f_dim / 2.f )
						.set_size( f_dim );

			shader.bbox = Box3D{ { 0, 0, 0 }, f_dim };
			shader.step = 1.f / lvl0_blksz;

			return true;
		}

	public:
		template <typename T>
		ThumbnailTexture<T> create_texture( std::shared_ptr<vol::Thumbnail<T>> const &thumb ) const
		{
			auto opts = Texture3DOptions{}
						  .set_device( device )
						  .set_dim( thumb->dim.x, thumb->dim.y, thumb->dim.z )
						  .set_opts( cufx::Texture::Options::as_array()
									   .set_address_mode( cufx::Texture::AddressMode::Clamp ) );
			ThumbnailTexture<T> texture( opts );
			if ( !device.has_value() ) { texture.thumb = thumb; }
			texture.source( thumb->data() );
			return texture;
		}

		Image<typename Shader::Pixel> create_film() const
		{
			auto img_opts = ImageOptions{}
							  .set_device( device )
							  .set_resolution( resolution );
			return Image<typename Shader::Pixel>( img_opts );
		}

	protected:
		vm::Option<cufx::Device> device;
		vec3 clear_color;
		uvec3 dim;
		Shader shader;
		Exhibit exhibit;
	};
}

VM_END_MODULE()
