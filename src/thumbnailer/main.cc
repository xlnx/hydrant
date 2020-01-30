#include <string>
#include <cxxopts.hpp>
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

	Thumbnail<float> thumbnail( unarchiver.dim() );
	Statistics stats;
	StatisticsCollector collector( unarchiver );
	for ( int z = 0; z != unarchiver.dim().z; ++z ) {
		for ( int y = 0; y != unarchiver.dim().y; ++y ) {
			for ( int x = 0; x != unarchiver.dim().x; ++x ) {
				Idx idx = { x, y, z };
				collector.compute_into( idx, stats );
				if ( stats.src.max < max_t && stats.src.avg < avg_t ) {
					thumbnail[ idx ] = 0;
					// continue;
				} else {
					thumbnail[ idx ] = stats.src.avg;
				}
				// vm::println( "{} {} {} -> {} {}", x, y, z, stats.src.max, stats.src.avg );
			}
		}
	}

	ofstream os( in + ".thumb" );
	UnboundedStreamWriter writer( os );
	thumbnail.dump( writer );
}