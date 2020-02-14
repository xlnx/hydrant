#include <varch/utils/io.hpp>
#include <varch/unarchive/unarchiver.hpp>
#include <varch/thumbnail.hpp>
#include "blocks.hpp"

using namespace std;
using namespace vol;

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	bool BlocksRenderer::init( std::shared_ptr<Dataset> const &dataset,
							   RendererConfig const &cfg )
	{
		if ( !Super::init( dataset, cfg ) ) { return false; }

		auto my_cfg = cfg.params.get<BlocksRendererConfig>();
		shader.render_mode = my_cfg.mode == "volume" ? BrmVolume : BrmSolid;
		shader.density = my_cfg.density;

		image = CudaImage<typename Shader::Pixel>( cfg.resolution, device );

		auto &lvl0_arch = dataset->meta.sample_levels[ 0 ].archives[ 0 ];
		ifstream is( dataset->root.resolve( lvl0_arch.path ).resolved(), ios::ate | ios::binary );
		auto len = is.tellg();
		vol::StreamReader reader( is, 0, len );
		vol::Unarchiver unarchiver( reader );

		glm::uvec3 dim = { unarchiver.dim().x, unarchiver.dim().y, unarchiver.dim().z };
		glm::uvec3 bdim = { unarchiver.padded_block_size(), unarchiver.padded_block_size(), unarchiver.padded_block_size() };

		glm::vec3 raw = { unarchiver.raw().x, unarchiver.raw().y, unarchiver.raw().z };
		glm::vec3 f_dim = raw / float( unarchiver.block_size() );
		// glm::vec3 max = { dim.x, dim.y, dim.z };
		exhibit = Exhibit{}
					.set_center( f_dim / 2.f )
					.set_size( f_dim );

		shader.bbox = Box3D{ { 0, 0, 0 }, f_dim };
		shader.step = 1e-2f * f_dim.x / 4.f;

		Thumbnail chebyshev_thumb( dataset->root.resolve( lvl0_arch.thumbnails[ "chebyshev" ] ).resolved() );
		chebyshev = ConstTexture3D<float>(
		  dim, chebyshev_thumb.data(),
		  cufx::Texture::Options::as_array()
			.set_address_mode( cufx::Texture::AddressMode::Clamp ),
		  device );
		shader.chebyshev_tex = chebyshev.value().get();

		Thumbnail mean_thumb( dataset->root.resolve( lvl0_arch.thumbnails[ "mean" ] ).resolved() );
		mean = ConstTexture3D<float>(
		  dim, mean_thumb.data(),
		  cufx::Texture::Options::as_array()
			.set_address_mode( cufx::Texture::AddressMode::Clamp ),
		  device );
		shader.mean_tex = mean.value().get();

		return true;
	}

	void BlocksRenderer::offline_render( std::string const &dst_path,
										 Camera const &camera )
	{
		raycaster.cast( exhibit, camera, image.value().view(), shader );
		image.value().view().copy_from_device().launch();
		image.value().get().dump( dst_path );
	}
}

VM_END_MODULE()
