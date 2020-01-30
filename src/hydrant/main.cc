#include <fstream>
#include <atomic>
#include <VMUtils/timer.hpp>
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

	Thumbnail<ThumbUnit> thumbnail( reader );

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

	auto th_4 = thumbnail.dim.x / 4.f;
	cufx::Image<Pixel> image( 512, 512 );
	Raycaster raycaster;
	{
		std::atomic_uint64_t total_steps( 0 );
		vm::Timer::Scoped timer( [&]( auto dt ) {
			vm::println( "time: {}   avg_step: {}",
						 dt.ms(), total_steps.load() / image.get_width() / image.get_height() );
		} );

		raycaster.cast(
		  exhibit, camera, image,
		  [&]( Ray const &ray ) -> Pixel {
			  const auto nsteps = 500;
			  const auto step = 1e-2f * th_4;
			  const auto opacity_threshold = 0.95f;
			  const auto density = 3e-3f;
			  const auto cdu = 1.f / std::max( std::abs( ray.d.x ),
											   std::max( std::abs( ray.d.y ),
														 std::abs( ray.d.z ) ) );

			  Pixel pixel = {};
			  float tnear, tfar;
			  if ( ray.intersect( bbox, tnear, tfar ) ) {
				  auto p = ray.o + ray.d * tnear;
				  int i;
				  for ( i = 0; i < nsteps; ++i ) {
					  p += ray.d * step;
					  glm::vec<3, int> pt = floor( p );

					  if ( !( pt.x >= 0 && pt.y >= 0 && pt.z >= 0 &&
							  pt.x < thumbnail.dim.x && pt.y < thumbnail.dim.y && pt.z < thumbnail.dim.z ) ) {
						  break;
					  }
					  if ( float cd = thumbnail[ { pt.x, pt.y, pt.z } ].chebyshev ) {
						  float tnear, tfar;
						  Ray{ p, ray.d }.intersect( Box3D{ pt, pt + 1 }, tnear, tfar );
						  auto d = tfar + ( cd - 1 ) * cdu;
						  i += d / step;
						  p += ray.d * d;
					  } else {
						  auto val = thumbnail[ { pt.x, pt.y, pt.z } ].value;
						  auto col = glm::vec4{ 1, 1, 1, 1 } * val * density;
						  pixel.v += col * ( 1.f - pixel.v.w );
						  if ( pixel.v.w > opacity_threshold ) break;
					  }
				  }
				  total_steps += i;
			  }
			  //   pixel.v = float4{ ray.d.x, ray.d.y, ray.d.z, 1 };
			  return pixel;
		  } );
	}

	image.dump( out );
}