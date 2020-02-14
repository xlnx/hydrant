#include <fstream>
#include <atomic>
#include <cstdlib>
#include <cppfs/fs.h>
#include <cppfs/FileHandle.h>
#include <cppfs/FilePath.h>
#include <VMUtils/fmt.hpp>
#include <VMUtils/timer.hpp>
#include <VMUtils/cmdline.hpp>
#include <hydrant/renderer.hpp>

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

int main( int argc, char **argv )
{
	cmdline::parser a;
	a.add<string>( "in", 'i', "input directory", true );
	a.add<string>( "out", 'o', "output filename", true );
	a.add( "thumb", 't', "take snapshots of single thumbnail file" );
	a.add<string>( "renderer", 'r', "renderer config file", false );
	a.add<string>( "camera", 'c', "camera config file", false );
	a.add<float>( "x", 'x', "camera.x", false, 3 );
	a.add<float>( "y", 'y', "camera.y", false, 2 );
	a.add<float>( "z", 'z', "camera.z", false, 2 );

	a.parse_check( argc, argv );

	auto in = FilePath( a.get<string>( "in" ) );
	ensure_dir( in.resolved() );
	auto out = FilePath( a.get<string>( "out" ) );
	auto device = cufx::Device::scan()[ 0 ];

	auto camera = Camera{};
	if ( a.exist( "camera" ) ) {
		auto cfg_path = FilePath( a.get<string>( "camera" ) );
		ensure_file( cfg_path.resolved() );
		camera = Camera::from_config( cfg_path.resolved() );
	} else {
		auto x = a.get<float>( "x" );
		auto y = a.get<float>( "y" );
		auto z = a.get<float>( "z" );
		camera.set_position( x, y, z );
	}

	RendererFactory factory( in );
	RendererConfig cfg;
	if ( a.exist( "renderer" ) ) {
		auto cfg_path = FilePath( a.get<string>( "renderer" ) );
		ensure_file( cfg_path.resolved() );
		ifstream is( cfg_path.resolved() );
		is >> cfg;
	}

	auto renderer = factory.create( cfg );
	renderer->offline_render( out.resolved(), camera );
}
