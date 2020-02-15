#pragma once

#include <varch/thumbnail.hpp>
#include <cudafx/device.hpp>
#include <hydrant/const_texture_3d.hpp>
#include <hydrant/renderer.hpp>

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	struct BasicRenderer;

	template <typename T>
	struct ThumbnailTexture : ConstTexture3D<T>
	{
		using ConstTexture3D<T>::ConstTexture3D;

	private:
		std::shared_ptr<vol::Thumbnail<T>> thumb;
		friend struct BasicRenderer;
	};

	struct BasicRendererConfig : vm::json::Serializable<BasicRendererConfig>
	{
		VM_JSON_FIELD( ShadingDevice, device ) = ShadingDevice::Cuda;
	};

	struct BasicRenderer : IRenderer
	{
		bool init( std::shared_ptr<Dataset> const &dataset,
				   RendererConfig const &cfg ) override
		{
			if ( !IRenderer::init( dataset, cfg ) ) { return false; }

			auto my_cfg = cfg.params.get<BasicRendererConfig>();
			if ( my_cfg.device == ShadingDevice::Cuda ) {
				device = cufx::Device::get_default();
				if ( !device.has_value() ) {
					vm::println( "cuda device not found, fallback to cpu render mode" );
				}
			}

			return true;
		}

	public:
		template <typename T>
		ThumbnailTexture<T> load_thumbnail( std::string const &path )
		{
			std::shared_ptr<vol::Thumbnail<T>> thumb( new vol::Thumbnail<T>( path ) );
			auto opts = ConstTexture3DOptions{}
						  .set_device( device )
						  .set_dim( thumb->dim.x, thumb->dim.y, thumb->dim.z )
						  .set_data( thumb->data() )
						  .set_opts( cufx::Texture::Options::as_array()
									   .set_address_mode( cufx::Texture::AddressMode::Clamp ) );
			ThumbnailTexture<T> texture( opts );
			if ( !device.has_value() ) { texture.thumb = thumb; }
			return texture;
		}

	protected:
		vm::Option<cufx::Device> device;
	};
}

VM_END_MODULE()
