#include <fstream>
#include <atomic>
#include <cstdlib>
#include <glog/logging.h>
#include <cppfs/fs.h>
#include <cppfs/FileHandle.h>
#include <cppfs/FilePath.h>
#include <VMUtils/fmt.hpp>
#include <VMUtils/timer.hpp>
#include <VMUtils/cmdline.hpp>
#include <hydrant/application.hpp>
#include <hydrant/core/renderer.hpp>

using namespace std;
using namespace vol;
using namespace hydrant;
using namespace cppfs;

inline void ensure_dir( std::string const &path_v )
{
	auto path = cppfs::fs::open( path_v );
	if ( !path.exists() ) {
		LOG( FATAL ) << vm::fmt( "the specified path '{}' doesn't exist",
								 path_v );
	} else if ( !path.isDirectory() ) {
		LOG( FATAL ) << vm::fmt( "the specified path '{}' is not a directory",
								 path_v );
	}
}

inline void ensure_file( std::string const &path_v )
{
	auto path = cppfs::fs::open( path_v );
	if ( !path.exists() ) {
		LOG( FATAL ) << vm::fmt( "the specified path '{}' doesn't exist",
								 path_v );
	} else if ( !path.isFile() ) {
		LOG( FATAL ) << vm::fmt( "the specified path '{}' is not a file",
								 path_v );
	}
}

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

	cmdline::parser a;
	a.add<string>( "in", 'i', "input directory", true );
	a.add<string>( "out", 'o', "output filename", false );
	a.add<string>( "config", 'c', "config file path", true );
	a.add<RealtimeRenderQuality>( "quality", 'q', "rt render quality", false,
								  RealtimeRenderQuality::Dynamic,
								  OptReader<RealtimeRenderQuality>() );
	a.add( "rt", 0, "real time render" );

	a.parse_check( argc, argv );

	auto in = FilePath( a.get<string>( "in" ) );
	ensure_dir( in.resolved() );

	auto cfg_path = FilePath( a.get<string>( "config" ) );
	ensure_file( cfg_path.resolved() );
	ifstream is( cfg_path.resolved() );
	Config cfg;
	is >> cfg;

	RendererFactory factory( in );
	auto renderer = factory.create( cfg.render );

	if ( !a.exist( "rt" ) ) {
		auto out = FilePath( a.get<string>( "out" ) );

		vm::Timer::Scoped _( []( auto dt ) {
			vm::println( "time: {}", dt.ms() );
		} );

		renderer->offline_render( cfg.camera ).dump( out.resolved() );
	} else {
		// auto opts = GlfwRenderLoopOptions{}
		// 			  .set_resolution( 1600, 900 )
		// 			  .set_title( "hydrant" );
		// LocalRenderLoop loop( opts, cfg, *renderer );
		// loop.orbit = *cfg.camera.orbit;

		Application app( factory, { cfg } );
		app.run();

		// renderer->realtime_render(
		//   loop,
		//   RealtimeRenderOptions{}
		// 	.set_quality( a.get<RealtimeRenderQuality>( "quality" ) ) );
	}
}
