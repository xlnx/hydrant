#include <string>
#include <cxxopts.hpp>
#include <VMUtils/timer.hpp>
#include <varch/utils/unbounded_io.hpp>
#include <varch/unarchive/unarchiver.hpp>
#include <varch/unarchive/statistics.hpp>
#include <thumbnail.hpp>

using namespace std;
using namespace vol;
using namespace hydrant;

int main( int argc, char **argv )
{
	cxxopts::Options options( "thumbnailer", "create thumbnail file for archived volume data" );
	options.add_options()( "i,input", "input archive file name", cxxopts::value<string>() );

	auto opts = options.parse( argc, argv );
	auto in = opts[ "i" ].as<string>();

	ifstream is( in, ios::ate | ios::binary );
	auto len = is.tellg();
	StreamReader reader( is, 0, len );
	Unarchiver unarchiver( reader );

	const double max_t = 8, avg_t = 1e-3;

	Thumbnail<ThumbUnit> thumbnail( unarchiver.dim() );
	Statistics stats;
	StatisticsCollector collector( unarchiver );

	{
		vm::Timer::Scoped timer(
		  [&]( auto dt ) {
			  vm::println( "thumbnailing time: {}", dt.ms() );
		  } );
		thumbnail.iterate_3d(
		  [&]( Idx const &idx ) {
			  collector.compute_into( idx, stats );
			  if ( stats.src.max < max_t && stats.src.avg < avg_t ) {
				  thumbnail[ idx ].value = 0;
				  // continue;
			  } else {
				  thumbnail[ idx ].value = stats.src.avg;
			  }
		  } );
	}

	{
		vm::Timer::Scoped timer(
		  [&]( auto dt ) {
			  vm::println( "chebyshev compute time: {}", dt.ms() );
		  } );
		thumbnail.compute_chebyshev();
	}

	// thumbnail.iterate_3d(
	//   [&]( Idx const &idx ) {
	// 	  vm::println( "{} -> {}", idx, thumbnail[ idx ].chebyshev );
	//   } );

	ofstream os( in + ".thumb" );
	UnboundedStreamWriter writer( os );
	thumbnail.dump( writer );
}