#include <fstream>
#include <atomic>
#include <cstdlib>
#include <cppfs/fs.h>
#include <cppfs/FileHandle.h>
#include <cppfs/FilePath.h>
#include <VMUtils/fmt.hpp>
#include <VMUtils/timer.hpp>
#include <VMUtils/cmdline.hpp>
#include <hydrant/core/renderer.hpp>

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
	} else if ( !path.isDirectory() ) {
		vm::eprintln( "the specified path '{}' is not a directory",
					  path_v );
		exit( 1 );
	}
}

inline void ensure_file( std::string const &path_v )
{
	auto path = cppfs::fs::open( path_v );
	if ( !path.exists() ) {
		vm::eprintln( "the specified path '{}' doesn't exist",
					  path_v );
		exit( 1 );
	} else if ( !path.isFile() ) {
		vm::eprintln( "the specified path '{}' is not a file",
					  path_v );
		exit( 1 );
	}
}

struct Config : vm::json::Serializable<Config>
{
	VM_JSON_FIELD( CameraConfig, camera );
	VM_JSON_FIELD( RendererConfig, render );
};

int main( int argc, char **argv )
{
	cmdline::parser a;
	a.add<string>( "in", 'i', "input directory", true );
	a.add<string>( "out", 'o', "output filename", true );
	a.add<string>( "config", 'c', "config file path", true );

	a.parse_check( argc, argv );

	auto in = FilePath( a.get<string>( "in" ) );
	ensure_dir( in.resolved() );
	auto out = FilePath( a.get<string>( "out" ) );

	auto cfg_path = FilePath( a.get<string>( "config" ) );
	ensure_file( cfg_path.resolved() );
	ifstream is( cfg_path.resolved() );
	Config cfg;
	is >> cfg;

	RendererFactory factory( in );
	auto renderer = factory.create( cfg.render );
	auto camera = Camera::from_config( cfg.camera );

	{
		vm::Timer::Scoped _( []( auto dt ) {
			vm::println( "time: {}", dt.ms() );
		} );

		renderer->offline_render( out.resolved(), camera );
	}
}
