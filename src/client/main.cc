#include <fstream>
#include <atomic>
#include <cstdlib>
#include <glog/logging.h>
#include <VMUtils/fmt.hpp>
#include <VMUtils/timer.hpp>
#include <VMUtils/cmdline.hpp>
#include "client.hpp"

using namespace std;
// using namespace vol;
using namespace hydrant;

template <typename Enum>
struct OptReader
{
	Enum operator()( const std::string &str )
	{
		return Enum::_from_string( str.c_str() );
	}
};

int main( int argc, char **argv )
{
	google::InitGoogleLogging( argv[ 0 ] );
	google::SetStderrLogging( google::GLOG_INFO );

	cmdline::parser a;
	a.add<string>( "server", '\0', "render server address, <host>:<port>", false, "localhost:9002" );
	a.add<string>( "in", 'i', "input directory", true );
	a.add<string>( "config", 'c', "config file path", true );
	a.add<RealtimeRenderQuality>( "quality", 'q', "rt render quality", false,
								  RealtimeRenderQuality::Dynamic,
								  OptReader<RealtimeRenderQuality>() );
	a.add( "rt", 0, "real time render" );

	a.parse_check( argc, argv );

	auto srv_addr = "ws://" + a.get<string>( "server" );

	auto cfg = Config{}.set_data_path( a.get<string>( "in" ) );
	std::ifstream is( a.get<string>( "config" ) );
	is >> cfg.params;

	Client clt( srv_addr, cfg );
	clt.run();
}
