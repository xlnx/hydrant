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

using namespace hydrant;

int main()
{
	cufx::Image<Pixel> image( 512, 512 );
	// image.at( 0, 0 ).v = { 0, 0, 1 };
	// image.dump( "test.png" );
	glm::vec3 min = { 0, 0, 0 };
	glm::vec3 max = { 2, 3, 4 };
	auto exhibit = Exhibit{}
					 .set_center( 1, 1.5, 2 )
					 .set_size( max );

	auto bbox = Box3D{ min, max };

	auto camera = Camera{}
					.set_position( 2, 1, 0 );
	Raycaster raycaster;
	raycaster.cast(
	  exhibit, camera, image,
	  [&]( Ray const &ray ) -> Pixel {
		  Pixel pixel;
		  float tnear, tfar;
		  if ( ray.intersect( bbox, tnear, tfar ) ) {
			  auto v = ( ray.o + ray.d * tnear ) / 4.f;
			  pixel.v = glm::vec4{ v.x, v.y, v.z, 1 };
		  } else {
			  pixel.v = glm::vec4{ 0, 0, 0, 1 };
		  }
		  //   pixel.v = float4{ ray.d.x, ray.d.y, ray.d.z, 1 };
		  return pixel;
	  } );

	image.dump( "test.png" );
}