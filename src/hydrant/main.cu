#include <fstream>
#include <atomic>
#include <VMUtils/timer.hpp>
#include <VMUtils/cmdline.hpp>
#include <varch/utils/io.hpp>
#include <varch/unarchive/unarchiver.hpp>
#include <cudafx/transfer.hpp>
#include <thumbnail.hpp>
#include <texture3d.hpp>
#include <buffer3d.hpp>
#include "shaders/scratch.hpp"
#include "shaders/volume_shader.hpp"

using namespace std;
using namespace vol;
using namespace hydrant;

int main( int argc, char **argv )
{
	cmdline::parser a;
	a.add<string>( "in", 'i', "input filename", true );
	a.add<string>( "out", 'o', "output filename", true );
	a.add( "thumb", 't', "take snapshots of single thumbnail file" );
	a.add<string>( "config", 'c', "config file", false );
	a.add<float>( "x", 'x', "camera.x", false, 3 );
	a.add<float>( "y", 'y', "camera.y", false, 2 );
	a.add<float>( "z", 'z', "camera.z", false, 2 );

	a.parse_check( argc, argv );

	auto in = a.get<string>( "in" );
	auto out = a.get<string>( "out" );
	auto device = cufx::Device::scan()[ 0 ];

	using Shader = VolumnRayEmitShader;
	Shader shader;

	cufx::Image<typename Shader::Pixel> image( 512, 512 );
	auto device_swap = device.alloc_image_swap( image );
	auto img_view = image.view().with_device_memory( device_swap.second );
	img_view.copy_to_device().launch();

	/* input file */

	ifstream is( in, ios::ate | ios::binary );
	auto len = is.tellg();
	StreamReader reader( is, 0, len );
	Unarchiver unarchiver( reader );
	Thumbnail<ThumbUnit> thumbnail( in + ".thumb" );

	glm::uvec3 dim = { unarchiver.dim().x, unarchiver.dim().y, unarchiver.dim().z };
	glm::uvec3 bdim = { unarchiver.block_size(), unarchiver.block_size(), unarchiver.block_size() };

	/* thumbnail texture */

	Texture3D<float2> thumbnail_texture(
	  device, reinterpret_cast<float2*>( thumbnail.data() ), dim,
	  cufx::Texture::Options::as_array() );
	shader.thumbnail_tex = thumbnail_texture.get();

	/* cache texture */

	auto props = device.props();
	glm::uvec3 tex3d_max_dim = { props.maxTexture3D[ 0 ],
								 props.maxTexture3D[ 1 ],
								 props.maxTexture3D[ 2 ] };
	Buffer3D<uint3> cache_buf( tex3d_max_dim / bdim );
	Texture3D<uint3> cache_texture(
	  device, cache_buf.data(), cache_buf.dim(),
	  cufx::Texture::Options::as_array() );
	shader.cache_tex = cache_texture.get();

	/* absent buffer */

	auto wg_cnt = 32 * 32;
	shader.wg_max_emit_cnt = 8;
	shader.wg_len_bytes = sizeof( int ) +
						  shader.wg_max_emit_cnt * sizeof( glm::uvec3 );
	auto absent_glob = device.alloc_global( shader.wg_len_bytes * wg_cnt );
	vector<char> absent( absent_glob.size() );
	shader.absent_buf = absent_glob.view_1d<char>( absent_glob.size() );

	/* block buffer */

	auto block_size = unarchiver.block_size();
	auto block_bytes = block_size * block_size * block_size;
	auto block_glob = device.alloc_global( block_bytes );
	auto block_view_1d = block_glob.view_1d<unsigned char>( block_bytes );
	auto block_view_info = cufx::MemoryView2DInfo{}
							 .set_stride( block_size * sizeof( unsigned char ) )
							 .set_width( block_size )
							 .set_height( block_size );
	auto block_extent = cufx::Extent{}
						  .set_width( block_size )
						  .set_height( block_size )
						  .set_depth( block_size );
	auto block_view_3d = block_glob.view_3d<unsigned char>( block_view_info, block_extent );

	auto sampler_extent = cufx::Extent{}
							.set_width( block_size )
							.set_height( block_size )
							.set_depth( block_size );
	auto sampler_arr = device.alloc_arraynd<unsigned char, 3>( sampler_extent );
	// cufx::MemoryView3D<float2> thumbnail_view( thumbnail.data(), thumbnail_view_info, thumbnail_extent );
	cufx::memory_transfer( sampler_arr, block_view_3d, cudaPos{ 0, 0, 0 } ).launch();
	cufx::Texture sampler_texture( sampler_arr, cufx::Texture::Options::as_array() );

	/* exhibit */

	glm::vec3 min = { 0, 0, 0 };
	glm::vec3 max = { dim.x, dim.y, dim.z };
	auto exhibit = Exhibit{}
					 .set_center( max / 2.f )
					 .set_size( max );

	shader.bbox = Box3D{ min, max };
	shader.th_4 = dim.x / 4.f;

	/* camera */

	auto camera = Camera{};
	if ( a.exist( "config" ) ) {
		auto cfg = a.get<string>( "config" );
		camera = Camera::from_config( cfg );
	} else {
		auto x = a.get<float>( "x" );
		auto y = a.get<float>( "y" );
		auto z = a.get<float>( "z" );
		camera.set_position( x, y, z );
	}

	Raycaster raycaster;
	{
		std::atomic_uint64_t total_steps( 0 );
		vm::Timer::Scoped timer( [&]( auto dt ) {
			vm::println( "time: {}   avg_step: {}",
						 dt.ms(), total_steps.load() / image.get_width() / image.get_height() );
		} );

		raycaster.cast( exhibit, camera, img_view, shader );

		cufx::memory_transfer( cufx::MemoryView1D<char>( absent ), shader.absent_buf ).launch();
		vector<Idx> block_idxs;

		for ( int i = 0; i != wg_cnt; ++i ) {
			auto wg_base_ptr = absent.data() + i * shader.wg_len_bytes;
			int wg_emit_cnt = *(int *)wg_base_ptr;
			glm::uvec3 *wg_ptr = (glm::uvec3 *)( wg_base_ptr + sizeof( int ) );
			for ( int j = 0; j != wg_emit_cnt; ++j ) {
				if ( glm::all( glm::lessThan( wg_ptr[ j ], dim ) ) ) {
					block_idxs.emplace_back( Idx{}
											   .set_x( wg_ptr[ j ].x )
											   .set_y( wg_ptr[ j ].y )
											   .set_z( wg_ptr[ j ].z ) );
				}
			}
			// if ( wg_emit_cnt ) {
			// 	vm::println( "#{} : wg_emit_cnt = {}", i, wg_emit_cnt );
			// 		vm::println( "{}", wg_ptr[ j ] );
			// 	}
			// }
		}
		std::sort( block_idxs.begin(), block_idxs.end() );
		auto last = std::unique( block_idxs.begin(), block_idxs.end() );
		block_idxs.erase( last, block_idxs.end() );

		unsigned nbytes = 0;
		unarchiver.unarchive(
		  block_idxs,
		  [&]( Idx const &idx, VoxelStreamPacket const &pkt ) {
			  pkt.append_to( block_view_1d );
			  nbytes += pkt.length;
			  if ( nbytes >= block_bytes ) {
				  // vm::println( "done {}", idx );
				  nbytes = 0;
			  }
		  } );
		// unarchiver.unarchive( block_idxs, []( Idx const &idx, VoxelStreamPacket const &) {} );
	}

	img_view.copy_from_device().launch();

	image.dump( out );
}
