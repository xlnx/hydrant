#include <fstream>
#include <atomic>
#include <VMUtils/timer.hpp>
#include <VMUtils/cmdline.hpp>
#include <varch/utils/io.hpp>
#include <thumbnail.hpp>
#include "shaders/scratch.hpp"

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

	ifstream is( in, ios::ate | ios::binary );
	auto len = is.tellg();
	StreamReader reader( is, 0, len );

	using Shader = ChebyshevShader<ScratchIntegrator>;

	Thumbnail<ThumbUnit> thumbnail( reader );
	Shader shader;

	glm::vec3 min = { 0, 0, 0 };
	glm::vec3 max = { thumbnail.dim.x, thumbnail.dim.y, thumbnail.dim.z };
	auto exhibit = Exhibit{}
					 .set_center( max / 2.f )
					 .set_size( max );

	shader.bbox = Box3D{ min, max };
	shader.th_4 = thumbnail.dim.x / 4.f;
	thumbnail.iterate_3d(
	  [&]( Idx const &idx ) {
		  auto &dst = shader.thumbnail[ idx.x ][ idx.y ][ idx.z ];
		  auto &src = thumbnail[ idx ];
		  dst = { src.value, src.chebyshev };
	  } );

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

	auto devices = cufx::Device::scan();
	cufx::Image<typename Shader::Pixel> image( 512, 512 );
	auto device_swap = devices[ 0 ].alloc_image_swap( image );
	auto img_view = image.view().with_device_memory( device_swap.second );
	img_view.copy_to_device().launch();

	Raycaster raycaster;
	{
		std::atomic_uint64_t total_steps( 0 );
		vm::Timer::Scoped timer( [&]( auto dt ) {
			vm::println( "time: {}   avg_step: {}",
						 dt.ms(), total_steps.load() / image.get_width() / image.get_height() );
		} );

		raycaster.cast( exhibit, camera, img_view, shader );
	}

	img_view.copy_from_device().launch();

	image.dump( out );
}
