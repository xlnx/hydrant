#include <fstream>
#include <atomic>
#include <cstdlib>
#include <cppfs/fs.h>
#include <cppfs/FileHandle.h>
#include <cppfs/FilePath.h>
#include <VMUtils/timer.hpp>
#include <VMUtils/cmdline.hpp>
#include <VMUtils/fmt.hpp>
#include <cudafx/device.hpp>
#include <cudafx/transfer.hpp>
#include <varch/utils/io.hpp>
#include <varch/unarchive/unarchiver.hpp>
#include <varch/thumbnail.hpp>
#include <varch/package_meta.hpp>
#include <hydrant/buffer3d.hpp>
#include <hydrant/raycaster.hpp>
#include "shaders/volume_shader.hpp"

using namespace std;
using namespace vol;
using namespace hydrant;
using namespace cppfs;

inline void ensure_dir( std::string const &path_v )
{
	auto path = cppfs::fs::open( path_v );
	if ( !path.exists() ) {
		vm::eprintln( "the specified path '{}' doesn't exist",
					  path_v );
		exit( 1 );
	} else if ( path.isFile() ) {
		vm::eprintln( "the specified path '{}' is not a file",
					  path_v );
		exit( 1 );
	}
}

int main( int argc, char **argv )
{
	cmdline::parser a;
	a.add<string>( "in", 'i', "input directory", true );
	a.add<string>( "out", 'o', "output filename", true );
	a.add( "thumb", 't', "take snapshots of single thumbnail file" );
	a.add<string>( "config", 'c', "config file", false );
	a.add<float>( "x", 'x', "camera.x", false, 3 );
	a.add<float>( "y", 'y', "camera.y", false, 2 );
	a.add<float>( "z", 'z', "camera.z", false, 2 );

	a.parse_check( argc, argv );

	auto in = FilePath( a.get<string>( "in" ) );
	ensure_dir( in.resolved() );
	auto out = FilePath( a.get<string>( "out" ) );
	auto device = cufx::Device::scan()[ 0 ];

	PackageMeta meta;
	ifstream meta_is( in.resolve( "package_meta.json" ).resolved() );
	meta_is >> meta;

	using Shader = VolumeRayEmitShader;
	Shader shader;

	cufx::Image<typename Shader::Pixel> image( 512, 512 );
	auto device_swap = device.alloc_image_swap( image );
	auto img_view = image.view().with_device_memory( device_swap.second );
	img_view.copy_to_device().launch();

	/* input file */

	auto &lvl0_arch = meta.sample_levels[ 0 ].archives[ 0 ];
	ifstream is( in.resolve( lvl0_arch.path ).resolved(), ios::ate | ios::binary );
	auto len = is.tellg();
	StreamReader reader( is, 0, len );
	Unarchiver unarchiver( reader );
	Thumbnail chebyshev( in.resolve( lvl0_arch.thumbnails[ "chebyshev" ] ).resolved() );

	glm::uvec3 dim = { unarchiver.dim().x, unarchiver.dim().y, unarchiver.dim().z };
	glm::uvec3 bdim = { unarchiver.padded_block_size(), unarchiver.padded_block_size(), unarchiver.padded_block_size() };

	/* view */
#pragma region

	glm::vec3 raw = { unarchiver.raw().x, unarchiver.raw().y, unarchiver.raw().z };
	glm::vec3 f_dim = raw / float( unarchiver.block_size() );
	// glm::vec3 max = { dim.x, dim.y, dim.z };
	auto exhibit = Exhibit{}
					 .set_center( f_dim / 2.f )
					 .set_size( f_dim );

	shader.bbox = Box3D{ { 0, 0, 0 }, f_dim };
	shader.step = 1e-2f * f_dim.x / 4.f;
	shader.cache_du.x = float( unarchiver.padding() ) / unarchiver.block_size();
	shader.cache_du.y = float( unarchiver.block_size() ) / unarchiver.padded_block_size();

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

#pragma endregion

	/* chebyshev texture */
#pragma region

	auto thumbnail_extent = cufx::Extent{}
							  .set_width( dim.x )
							  .set_height( dim.y )
							  .set_depth( dim.z );
	auto thumbnail_view_info = cufx::MemoryView2DInfo{}
								 .set_stride( dim.x * sizeof( float ) )
								 .set_width( dim.x )
								 .set_height( dim.y );
	auto chebyshev_arr = device.alloc_arraynd<float, 3>( thumbnail_extent );
	// vm::println( "dim = {}, thumbnail_extent = {}", dim, thumbnail_extent );
	cufx::MemoryView3D<float> chebyshev_view( chebyshev.data(), thumbnail_view_info, thumbnail_extent );
	cufx::memory_transfer( chebyshev_arr, chebyshev_view ).launch();
	cufx::Texture chebyshev_texture( chebyshev_arr,
									 cufx::Texture::Options::as_array()
									   .set_address_mode( cufx::Texture::AddressMode::Clamp ) );
	shader.chebyshev_tex = chebyshev_texture;

#pragma endregion

	/* present texture */
#pragma region

	Buffer3D<int> present_buf( dim );
	auto present_extent = cufx::Extent{}
							.set_width( dim.x )
							.set_height( dim.y )
							.set_depth( dim.z );
	auto present_arr = device.alloc_arraynd<int, 3>( present_extent );
	// vm::println( "dim = {}, present_extent = {}", dim, present_extent );
	auto present_view_info = cufx::MemoryView2DInfo{}
							   .set_stride( dim.x * sizeof( int ) )
							   .set_width( dim.x )
							   .set_height( dim.y );
	cufx::MemoryView3D<int> present_view( present_buf.data(), present_view_info, present_extent );
	// cufx::Texture present_texture( present_arr, cufx::Texture::Options::as_array() );
	// cufx::memory_transfer( present_arr, present_view ).launch();
	// cufx::Texture present_texture( present_arr, cufx::Texture::Options::as_array() );
	// shader.present_tex = present_texture;

#pragma endregion

	/* absent buffer */
#pragma region

	auto wg_cnt = 32 * 32;
	shader.wg_max_emit_cnt = 8;
	shader.wg_len_bytes = sizeof( int ) +
						  shader.wg_max_emit_cnt * sizeof( glm::uvec3 );
	auto absent_glob = device.alloc_global( shader.wg_len_bytes * wg_cnt );
	vector<char> absent( absent_glob.size() );
	shader.absent_buf = absent_glob.view_1d<char>( absent_glob.size() );

#pragma endregion

	/* block buffer */

	auto pad_bs = unarchiver.padded_block_size();
	auto block_bytes = pad_bs * pad_bs * pad_bs;
	auto block_glob = device.alloc_global( block_bytes );
	auto block_view_1d = block_glob.view_1d<unsigned char>( block_bytes );
	auto block_view_info = cufx::MemoryView2DInfo{}
							 .set_stride( pad_bs * sizeof( unsigned char ) )
							 .set_width( pad_bs )
							 .set_height( pad_bs );
	auto block_extent = cufx::Extent{}
						  .set_width( pad_bs )
						  .set_height( pad_bs )
						  .set_depth( pad_bs );
	auto block_view_3d = block_glob.view_3d<unsigned char>( block_view_info, block_extent );
	vector<cufx::Array3D<unsigned char>> cache_block_arr;
	for ( int i = 0; i != MAX_CACHE_SIZE; ++i ) {
		cache_block_arr.emplace_back( device.alloc_arraynd<unsigned char, 3>( block_extent ) );
	}
	// cufx::MemoryView3D<int> chebyshev_view( chebyshev.data(), thumbnail_view_info, thumbnail_extent );
	// cufx::memory_transfer( sampler_arr, block_view_3d, cudaPos{ 0, 0, 0 } ).launch();
	// cufx::Texture sampler_texture( sampler_arr, cufx::Texture::Options::as_array() );

	std::vector<vol::Idx> block_idxs;
	chebyshev.iterate_3d(
	  [&]( vol::Idx const &idx ) {
		  if ( !chebyshev[ idx ] ) {
			  block_idxs.emplace_back( idx );
		  }
	  } );
	vm::println( "{}", block_idxs );
	vector<glm::vec3> block_ccs( block_idxs.size() );
	std::transform( block_idxs.begin(), block_idxs.end(), block_ccs.begin(),
					[]( Idx const &idx ) { return glm::vec3( idx.x, idx.y, idx.z ) + 0.5f; } );
	vector<int> pidx( block_idxs.size() );
	for ( int i = 0; i != pidx.size(); ++i ) { pidx[ i ] = i; }
	vm::println( "{}", block_idxs.size() );

	auto et = exhibit.get_matrix();

	Raycaster raycaster;
	// raycaster.cast( exhibit, camera, img_view, shader );
	int nframes = 1;
	while ( nframes-- ) {
		std::size_t ns = 0, ns1 = 0;

		vm::Timer::Scoped timer( [&]( auto dt ) {
			vm::println( "time: {} / {} / {}", dt.ms(),
						 ns / 1000 / 1000,
						 ns1 / 1000 / 1000 );
		} );

		glm::vec3 cp = et * glm::vec4( camera.position, 1 );

		std::sort( pidx.begin(), pidx.end(),
				   [&]( int x, int y ) {
					   return glm::distance( block_ccs[ x ], cp ) <
							  glm::distance( block_ccs[ y ], cp );
				   } );

		for ( int i = 0; i < pidx.size(); i += MAX_CACHE_SIZE ) {
			vector<Idx> idxs;
			for ( int j = i; j < i + MAX_CACHE_SIZE && j < pidx.size(); ++j ) {
				idxs.emplace_back( block_idxs[ pidx[ j ] ] );
			}

			int nbytes = 0, blkid = 0;
			vector<cufx::Texture> cache_texs;
			memset( present_buf.data(), -1, present_buf.bytes() );

			{
				vm::Timer::Scoped timer( [&]( auto dt ) {
					ns += dt.ns().cnt();
				} );

				unarchiver.unarchive(
				  idxs,
				  [&]( Idx const &idx, VoxelStreamPacket const &pkt ) {
					  pkt.append_to( block_view_1d );
					  nbytes += pkt.length;
					  if ( nbytes >= block_bytes ) {
						  cufx::memory_transfer( cache_block_arr[ blkid ], block_view_3d ).launch();
						  //   if ( blkid == 0 ) {
						  cache_texs.emplace_back( cache_block_arr[ blkid ],
												   cufx::Texture::Options{}
													 .set_address_mode( cufx::Texture::AddressMode::Wrap )
													 .set_filter_mode( cufx::Texture::FilterMode::Linear )
													 .set_read_mode( cufx::Texture::ReadMode::NormalizedFloat )
													 .set_normalize_coords( true ) );
						  present_buf[ glm::vec3( idx.x, idx.y, idx.z ) ] = blkid;
						  nbytes = 0;
						  blkid += 1;
						  //   }
					  }
				  } );
			}

			cufx::memory_transfer( present_arr, present_view ).launch();
			cufx::Texture present_texture( present_arr, cufx::Texture::Options::as_array() );
			shader.present_tex = present_texture;

			// vm::println( "{}", cache_texs.size() );
			for ( int j = 0; j != cache_texs.size(); ++j ) {
				shader.cache_tex[ j ] = cache_texs[ j ];
			}

			{
				vm::Timer::Scoped timer( [&]( auto dt ) {
					ns1 += dt.ns().cnt();
				} );

				if ( i == 0 ) {
					raycaster.cast( exhibit, camera, img_view, shader );
				} else {
					raycaster.cast( img_view, reinterpret_cast<VolumePixelShader &>( shader ) );
				}
			}
		}
	}

	img_view.copy_from_device().launch();

	image.dump( out.resolved() );
}
