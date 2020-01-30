#include <fstream>
#include <VMUtils/cmdline.hpp>
#include <varch/utils/io.hpp>
#include <thumbnail.hpp>
#include "raycaster.hpp"

struct Pixel
{
	void write_to( unsigned char dst[ 4 ] )
	{
		auto v = glm::clamp( this->v * 255.f,
							 glm::vec4{ 0, 0, 0, 0 },
							 glm::vec4{ 255, 255, 255, 255 } );
		dst[ 0 ] = (unsigned char)( v.x );
		dst[ 1 ] = (unsigned char)( v.y );
		dst[ 2 ] = (unsigned char)( v.z );
		dst[ 3 ] = (unsigned char)( 255 );
	}

public:
	glm::vec4 v;
	float t;
};

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

	Thumbnail<float> thumbnail( reader );

	glm::vec3 min = { 0, 0, 0 };
	glm::vec3 max = { thumbnail.dim.x, thumbnail.dim.y, thumbnail.dim.z };
	auto exhibit = Exhibit{}
					 .set_center( max / 2.f )
					 .set_size( max );
	auto bbox = Box3D{ min, max };

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

	cufx::Image<Pixel> image( 512, 512 );
	Raycaster raycaster;
	raycaster.cast(
	  exhibit, camera, image,
	  [&]( Ray const &ray ) -> Pixel {
		  const auto nsteps = 500;
		  const auto step = ray.d * 1e-2f;

		  Pixel pixel = {};
		  float tnear, tfar;
		  if ( ray.intersect( bbox, tnear, tfar ) ) {
			  auto p = ray.o + ray.d * tnear;
			  for ( int i = 0; i != nsteps; ++i ) {
				  p += step;
				  glm::vec<3, int> pt = p;
				  if ( pt.x >= 0 && pt.y >= 0 && pt.z >= 0 &&
					   pt.x < thumbnail.dim.x && pt.y < thumbnail.dim.y && pt.z < thumbnail.dim.z ) {
					  if ( thumbnail[ { pt.x, pt.y, pt.z } ] ) {
						  pixel.v = glm::vec4{ 1, 1, 1, 1 };
					  }
				  }
			  }
		  }
		  //   pixel.v = float4{ ray.d.x, ray.d.y, ray.d.z, 1 };
		  return pixel;
	  } );

	image.dump( out );
}